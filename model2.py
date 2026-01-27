import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAttentionTransformerNet(nn.Module):
    def __init__(
        self,
        window_length: int,
        feature_dim: int = 128,
        desc_dim: int = 18,
        state_dim: int = 7,
        nhead: int = 4,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        """
        window_length: number of timesteps W
        feature_dim:    latent size F
        desc_dim:       per-joint description output size (18)
        state_dim:      general policy state size (7)
        nhead:          number of attention heads
        num_layers:     number of Transformer layers in each block
        """
        super().__init__()
        self.F = feature_dim
        self.W = window_length
        self.state_dim = state_dim

        # per-(t,j) embedding: 7 → F
        self.embed = nn.Linear(7, self.F)
        self.act   = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        # temporal Transformer: attends across W for each joint
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.F, nhead=nhead, batch_first=False, dropout=dropout_rate
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # joint Transformer: attends across J after time pooling
        joint_layer = nn.TransformerEncoderLayer(
            d_model=self.F, nhead=nhead, batch_first=False, dropout=dropout_rate
        )
        self.joint_encoder = nn.TransformerEncoder(
            joint_layer, num_layers=num_layers
        )

        # output heads
        self.desc_head  = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.F, desc_dim)
        )
        self.state_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.F, state_dim)
        )
        
        # Hardcoded nominal joint range values for default robot (12 joints, 2 values: [lower, upper])
        nominal_joint_range_hard_coded = torch.tensor([
            [-1.0499998,  1.0499998],
            [-1.0499998,  1.0499998],
            [-1.0499998,  1.0499998],
            [-1.0499998,  1.0499998],
            [-1.5699999,  3.4899998],
            [-1.5699999,  3.4899998],
            [-0.5199999,  4.5299997],
            [-0.5199999,  4.5299997],
            [-2.7199998, -0.84     ],
            [-2.7199998, -0.84     ],
            [-2.7199998, -0.84     ],
            [-2.7199998, -0.84     ]
        ], dtype=torch.float32)
        
        # Register as buffer so it moves with model to correct device and is included in state_dict
        # self.register_buffer('nominal_joint_range_hard_coded', nominal_joint_range_hard_coded)

    def forward(
        self,
        dynamic_joint_state_seq: torch.Tensor,  # (B, W, J, 3)
        actions_seq:            torch.Tensor,   # (B, W, J, 1)
        command_seq:            torch.Tensor    # (B, W, 3)
    ):
        B, W, J, _ = dynamic_joint_state_seq.shape
        assert W == self.W, f"Expected W={self.W}, got {W}"

        # 1) embed per-(t,j)
        cmd = command_seq.unsqueeze(2).expand(B, W, J, 3)         # → (B,W,J,3)
        x   = torch.cat([dynamic_joint_state_seq,                # (B,W,J,3)
                         actions_seq,                            # (B,W,J,1)
                         cmd], dim=-1)                           # → (B,W,J,7)
        h   = self.act(self.embed(x))                            # → (B,W,J,F)
        h = self.dropout(h)

        # 2) temporal self-attention (per joint)
        #    flatten batch & joints into one big batch of size B*J
        h_t = h.permute(0,2,1,3).contiguous().view(B*J, W, self.F)  # (B*J, W, F)
        h_t = h_t.permute(1,0,2)                                    # (W, B*J, F)
        h_t = self.temporal_encoder(h_t)                           # (W, B*J, F)
        h_t = h_t.permute(1,0,2).view(B, J, W, self.F)              # (B, J, W, F)
        h   = h_t.permute(0,2,1,3)                                  # (B, W, J, F)

        # 3) pool across time → (B, J, F)
        z = h.mean(dim=1)

        # 4) joint self-attention
        z_j = z.permute(0,2,1).contiguous()      # (B, F, J)
        z_j = z_j.permute(2,0,1)                 # (J, B, F)
        z_j = self.joint_encoder(z_j)            # (J, B, F)
        z_j = z_j.permute(1,0,2)                 # (B, J, F)

        # 5) pool across joints → (B, F)
        s_inv = z_j.mean(dim=1)

        # 6) heads - predict residuals
        desc_residual = self.desc_head(z_j)   # (B, J, 18) - these are residuals
        state_residual = self.state_head(s_inv)  # (B, 7) - these are residuals

        # ========== RESIDUAL LEARNING: Add nominal joint range values at output ==========
        # Start with residuals, add nominal values for joint_range1 and joint_range2
        desc_pred = desc_residual.clone()
        state_pred = state_residual.clone()
        
        # Apply residual learning to joint range (indices 8 and 9)
        # Slice nominal_joint_range to match actual J from input
        # Buffer automatically on same device as model (no .to() needed)
        assert J == 12, f"Expected J=12, got {J}"
        # nominal_joint_range = self.nominal_joint_range_hard_coded  # (12, 2)
        # desc_pred[:, :, 9] = nominal_joint_range[:, 0].unsqueeze(0).expand(B, -1) + desc_residual[:, :, 9]
        # desc_pred[:, :, 10] = nominal_joint_range[:, 1].unsqueeze(0).expand(B, -1) + desc_residual[:, :, 10]

        dynamic_joint_description_pred = desc_residual
        invariant_general_policy_state_pred = state_residual

        return dynamic_joint_description_pred, invariant_general_policy_state_pred

def get_policy(window_length: int, model_device):
    policy = AdaptiveAttentionTransformerNet(window_length)
    policy = torch.jit.script(policy)
    policy.to(model_device)

    return policy

def main():
    # Example usage with randomized inputs
    B, W, J = 2, 20, 12
    net = AdaptiveAttentionTransformerNet(window_length=W)
    dynamic_joint_state_seq = torch.randn(B, W, J, 3)
    actions_seq = torch.randn(B, W, J, 1)
    command_seq = torch.randn(B, W, 3)

    desc_pred, state_pred = net(dynamic_joint_state_seq, actions_seq, command_seq)
    print("dynamic_joint_description_pred shape:", desc_pred.shape)
    print("invariant_general_policy_state_pred shape:", state_pred.shape)

if __name__ == "__main__":
    main()
