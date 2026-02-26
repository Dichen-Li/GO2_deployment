"""
Copied from silver_badger_torch/policy.py for standalone deployment.
Policy expects: dynamic_joint_description (J,18), dynamic_joint_state (J,3), general_state (10).
general_state = trunk(3) + goal(3) + gravity(3) + mass(1).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class Policy(nn.Module):
    def __init__(self, initial_softmax_temperature,
                 softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip,
                 policy_std_max_clip, args_cli=None):
        super(Policy, self).__init__()
        self.softmax_temperature_min = softmax_temperature_min
        self.stability_epsilon = stability_epsilon
        self.policy_mean_abs_clip = policy_mean_abs_clip
        self.policy_std_min_clip = policy_std_min_clip
        self.policy_std_max_clip = policy_std_max_clip
        self.args_cli = args_cli

        # Constants
        dynamic_joint_des_part1_dim = 14
        dynamic_joint_des_part2_dim = 4
        dynamic_joint_des_part2_latent_dim = 8
        dynamic_joint_des_original_dim = dynamic_joint_des_part1_dim + dynamic_joint_des_part2_dim
        dynamic_joint_des_mix_dim = dynamic_joint_des_part1_dim + dynamic_joint_des_part2_latent_dim
        general_state_dim = 10
        dynamic_joint_state_dim = 3

        self.register_buffer("idx1", torch.tensor([0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long))
        self.register_buffer("idx2", torch.tensor([6, 7, 9, 10], dtype=torch.long))

        self.dynamic_joint_des_part2_latent1 = nn.Linear(dynamic_joint_des_part2_dim, dynamic_joint_des_part2_latent_dim)
        self.dynamic_joint_des_part2_layer_norm = nn.LayerNorm(dynamic_joint_des_part2_latent_dim, eps=1e-6)
        self.dynamic_joint_des_part2_latent2 = nn.Linear(dynamic_joint_des_part2_latent_dim, dynamic_joint_des_part2_latent_dim)

        scale_factor = 1
        dynamic_joint_state_mask_dim = 128 * scale_factor
        dynamic_joint_state_feat = 4 * scale_factor
        self.dynamic_joint_state_mask1 = nn.Linear(dynamic_joint_des_mix_dim, dynamic_joint_state_mask_dim)
        self.dynamic_joint_layer_norm = nn.LayerNorm(dynamic_joint_state_mask_dim, eps=1e-6)
        self.dynamic_joint_state_mask2 = nn.Linear(dynamic_joint_state_mask_dim, dynamic_joint_state_mask_dim)
        self.joint_log_softmax_temperature = nn.Parameter(torch.tensor([initial_softmax_temperature - self.softmax_temperature_min]).log())
        self.latent_dynamic_joint_state = nn.Linear(dynamic_joint_state_dim, dynamic_joint_state_feat)

        combined_action_feat_dim = dynamic_joint_state_mask_dim * dynamic_joint_state_feat + general_state_dim
        action_latent_dims = [512, 256, 128 * scale_factor]
        self.action_latent1 = nn.Linear(combined_action_feat_dim, action_latent_dims[0])
        self.action_layer_norm = nn.LayerNorm(action_latent_dims[0], eps=1e-6)
        self.action_latent2 = nn.Linear(action_latent_dims[0], action_latent_dims[1])
        self.action_latent3 = nn.Linear(action_latent_dims[1], action_latent_dims[2])

        action_des_latent_dim = 128 * scale_factor
        self.action_description_latent1 = nn.Linear(dynamic_joint_des_mix_dim, action_des_latent_dim)
        self.action_description_layer_norm = nn.LayerNorm(action_des_latent_dim, eps=1e-6)
        self.action_description_latent2 = nn.Linear(action_des_latent_dim, action_des_latent_dim)

        policy_in_dim = dynamic_joint_state_feat + action_latent_dims[-1] + action_des_latent_dim
        policy_hidden_dim = 128 * scale_factor
        self.policy_mean_layer1 = nn.Linear(policy_in_dim, policy_hidden_dim)
        self.policy_mean_layer_norm = nn.LayerNorm(policy_hidden_dim, eps=1e-6)
        self.policy_mean_layer2 = nn.Linear(policy_hidden_dim, 1)
        self.policy_logstd_layer = nn.Linear(policy_hidden_dim, 1)

    def forward(self, dynamic_joint_description, dynamic_joint_state, general_state, latent_override=None):
        dynamic_joint_description_part1 = dynamic_joint_description[..., self.idx1]
        dynamic_joint_description_part2 = dynamic_joint_description[..., self.idx2]
        if latent_override is not None:
            dynamic_joint_description_part2_latent = latent_override
        else:
            dynamic_joint_description_part2_latent = self.dynamic_joint_des_part2_latent1(dynamic_joint_description_part2)
            dynamic_joint_description_part2_latent = F.elu(self.dynamic_joint_des_part2_layer_norm(dynamic_joint_description_part2_latent))
            dynamic_joint_description_part2_latent = torch.tanh(self.dynamic_joint_des_part2_latent2(dynamic_joint_description_part2_latent))

        dynamic_joint_description_mix = torch.cat([dynamic_joint_description_part1, dynamic_joint_description_part2_latent], dim=-1)
        dynamic_joint_state_mask = self.dynamic_joint_state_mask1(dynamic_joint_description_mix)
        dynamic_joint_state_mask = F.elu(self.dynamic_joint_layer_norm(dynamic_joint_state_mask))
        dynamic_joint_state_mask = torch.tanh(self.dynamic_joint_state_mask2(dynamic_joint_state_mask))
        dynamic_joint_state_mask = torch.clamp(dynamic_joint_state_mask,
                                               -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)

        latent_dynamic_joint_state = F.elu(self.latent_dynamic_joint_state(dynamic_joint_state))
        joint_e_x = torch.exp(dynamic_joint_state_mask / (torch.exp(self.joint_log_softmax_temperature) + self.softmax_temperature_min))
        dynamic_joint_state_mask = joint_e_x / (joint_e_x.sum(dim=-1, keepdim=True) + self.stability_epsilon)
        dynamic_joint_state_mask = dynamic_joint_state_mask.unsqueeze(-1).repeat(1, 1, 1, latent_dynamic_joint_state.size(-1))
        masked_dynamic_joint_state = dynamic_joint_state_mask * latent_dynamic_joint_state.unsqueeze(-2)
        masked_dynamic_joint_state = masked_dynamic_joint_state.view(masked_dynamic_joint_state.shape[:-2] + (masked_dynamic_joint_state.shape[-2] * masked_dynamic_joint_state.shape[-1],))
        dynamic_joint_latent = masked_dynamic_joint_state.sum(dim=-2)

        combined_input = torch.cat([dynamic_joint_latent, general_state], dim=-1)
        action_latent = self.action_latent1(combined_input)
        action_latent = F.elu(self.action_layer_norm(action_latent))
        action_latent = F.elu(self.action_latent2(action_latent))
        action_latent = self.action_latent3(action_latent)

        action_description_latent = self.action_description_latent1(dynamic_joint_description_mix)
        action_description_latent = F.elu(self.action_description_layer_norm(action_description_latent))
        action_description_latent = self.action_description_latent2(action_description_latent)

        action_latent = action_latent.unsqueeze(-2).repeat(1, action_description_latent.size(-2), 1)
        combined_action_latent = torch.cat([action_latent, latent_dynamic_joint_state.detach(), action_description_latent], dim=-1)

        policy_mean = self.policy_mean_layer1(combined_action_latent)
        policy_mean = F.elu(self.policy_mean_layer_norm(policy_mean))
        policy_mean = self.policy_mean_layer2(policy_mean)
        policy_mean = torch.clamp(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        return policy_mean.squeeze(-1)


def get_policy(model_device, args_cli=None):
    initial_softmax_temperature = 1.0
    softmax_temperature_min = 0.015
    stability_epsilon = 0.00000001
    policy_mean_abs_clip = 10.0
    policy_std_min_clip = 0.00000001
    policy_std_max_clip = 2.0

    policy = Policy(initial_softmax_temperature, softmax_temperature_min, stability_epsilon, policy_mean_abs_clip, policy_std_min_clip, policy_std_max_clip, args_cli)
    policy.to(model_device)
    return policy
