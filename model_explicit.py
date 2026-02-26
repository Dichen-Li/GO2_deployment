"""
Copied from adaptive_configure/model_explicit.py for standalone deployment.
Outputs: desc_pred (B, J, 4) for indices [6, 7, 9, 10], pol_pred (B, 1) for mass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAttentionTransformerNet(nn.Module):
    def __init__(
        self,
        window_length: int,
        feature_dim: int = 128,
        desc_dim: int = 4,  # joint_nominal_position, torque_limit, joint_range1, joint_range2
        state_dim: int = 1,  # mass only
        nhead: int = 4,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.F = feature_dim
        self.W = window_length
        self.state_dim = state_dim

        self.embed = nn.Linear(8, self.F)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.F, nhead=nhead, batch_first=False, dropout=dropout_rate
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        joint_layer = nn.TransformerEncoderLayer(
            d_model=self.F, nhead=nhead, batch_first=False, dropout=dropout_rate
        )
        self.joint_encoder = nn.TransformerEncoder(
            joint_layer, num_layers=num_layers
        )

        self.desc_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.F, desc_dim)
        )
        self.state_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.F, state_dim)
        )

        joint_nominal_positions = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,
                                                0.8000,  0.8000,  1.0000,  1.0000,
                                                -1.5000, -1.5000, -1.5000, -1.5000], dtype=torch.float32)
        self.register_buffer('joint_nominal_positions', joint_nominal_positions)

    def forward(
        self,
        dynamic_joint_state_seq: torch.Tensor,  # (B, W, J, 3)
        actions_seq: torch.Tensor,              # (B, W, J, 1)
        command_seq: torch.Tensor               # (B, W, 3)
    ):
        B, W, J, _ = dynamic_joint_state_seq.shape
        assert W == self.W, f"Expected W={self.W}, got {W}"

        cmd = command_seq.unsqueeze(2).expand(B, W, J, 3)
        jnp_seq = self.joint_nominal_positions.view(1, 1, J, 1).expand(B, W, J, 1)
        x = torch.cat([dynamic_joint_state_seq, actions_seq, cmd, jnp_seq], dim=-1)
        h = self.act(self.embed(x))
        h = self.dropout(h)

        h_t = h.permute(0, 2, 1, 3).contiguous().view(B * J, W, self.F)
        h_t = h_t.permute(1, 0, 2)
        h_t = self.temporal_encoder(h_t)
        h_t = h_t.permute(1, 0, 2).view(B, J, W, self.F)
        h = h_t.permute(0, 2, 1, 3)

        z = h.mean(dim=1)

        z_j = z.permute(0, 2, 1).contiguous()
        z_j = z_j.permute(2, 0, 1)
        z_j = self.joint_encoder(z_j)
        z_j = z_j.permute(1, 0, 2)

        desc_pred = self.desc_head(z_j)          # (B, J, 4)
        z_state = z_j.mean(dim=1)
        pol_pred = self.state_head(z_state)      # (B, 1)

        return desc_pred, pol_pred


def get_policy(window_length: int, model_device):
    policy = AdaptiveAttentionTransformerNet(window_length)
    policy = torch.jit.script(policy)
    policy.to(model_device)
    return policy
