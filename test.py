#!/usr/bin/env python3
"""
GO2 deployment with URMA expert policy + online adaptation injection (CPU).

Key requirements satisfied:
- RobotHandler remains the ONLY real-robot I/O interface (ROS2 subscriptions/publishing).
- Keep original deployment code behavior for:
  - observation retrieval
  - action sending (LowCmd publishing)
  - hard joint locking / clamping logic (kept exactly)
- Add adaptation with minimal intrusion:
  RobotHandler.nn() builds the same raw observation as before, then calls a runner:
    runner.step(obs) does:
      1) update_predictions (from past history)
      2) inject predictions (if ready)
      3) compute action
      4) add_history (state, action, cmd_vel) for next prediction
- No sim-style reset/respawn flags.

YOU MUST IMPLEMENT (placeholders):
- DeploymentAdaptationModule._build_model()
- DeploymentAdaptationModule.load_checkpoint(...)
- DeploymentAdaptationModule._forward_model(...)
"""
# MACRO DEFINITIO FOR DEPLOYMENT
JOINT_IDS_TO_LOCK = [3, 4, 5] # front left
HARD_LOCKED_FACTOR = 0.3
ADDED_MASS = 0
CMD_VEL_THRESHOLD = 0.0
"""
ENABLE_ADAPTATION = False                             -> default(blind)
ENABLE_ADAPTATION = True and CORRECT_..._FLAG = False -> adaptation
ENABLE_ADAPTATION = True and CORRECT_..._FLAG = True  -> correct(privilege)
"""
ENABLE_ADAPTATION = True
CORRECT_JR_FLAG = False
CORRECT_MASS_FLAG = False



import json
import os
import time
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from unitree_go.msg import LowState, LowCmd, WirelessController
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from crc_module import get_crc
from policy import get_policy


# -------------------------
# Deployment observation parser (standalone, matches simulation output format)
# Deployment obs: 262 elements. general_policy_state = trunk(3)+goal(3)+gravity(3)+mass(1) = 10 elements.
# Mass is at index 261 (indices 252-261 = trunk, goal, gravity, mass).
# -------------------------
def one_policy_observation_to_inputs(one_policy_observation: torch.Tensor, metadata: SimpleNamespace):
    """
    Parse deployment observation into policy inputs.
    Returns general_policy_state (10 elements) matching simulation format.
    """
    nr_dynamic_joint_observations = metadata.nr_dynamic_joint_observations
    single_dynamic_joint_observation_length = metadata.single_dynamic_joint_observation_length
    dynamic_joint_observation_length = metadata.dynamic_joint_observation_length
    dynamic_joint_description_size = metadata.dynamic_joint_description_size

    dynamic_joint_combined_state = one_policy_observation[..., :dynamic_joint_observation_length].view(
        (*one_policy_observation.shape[:-1], nr_dynamic_joint_observations, single_dynamic_joint_observation_length)
    )
    dynamic_joint_description = dynamic_joint_combined_state[..., :dynamic_joint_description_size]
    dynamic_joint_state = dynamic_joint_combined_state[..., dynamic_joint_description_size:]

    trunk_angular_vel_update_obs_idx = metadata.trunk_angular_vel_update_obs_idx
    goal_velocity_update_obs_idx = metadata.goal_velocity_update_obs_idx
    projected_gravity_update_obs_idx = metadata.projected_gravity_update_obs_idx
    # Deployment: trunk(252-254)+goal(255-257)+gravity(258-260), then mass(261)
    mass_idx = [projected_gravity_update_obs_idx[-1] + 1]
    general_policy_state = one_policy_observation[..., trunk_angular_vel_update_obs_idx+goal_velocity_update_obs_idx+projected_gravity_update_obs_idx+mass_idx]

    return dynamic_joint_description, dynamic_joint_state, general_policy_state


# ----------------------------------------
# Deployment Adaptation Module (single env)
# Uses model_explicit: desc_pred (B,J,4) for indices [6,7,9,10], pol_pred (B,1) for mass.
# ----------------------------------------
class DeploymentAdaptationModule:
    """
    Single-robot adaptation module matching eval_actor_urma_adaptive.py.
    """

    def __init__(self, window_length: int = 10, adaptation_freq: int = 1, num_joints: int = 12, device="cpu"):
        self.W = int(window_length)
        self.adaptation_freq = int(adaptation_freq)
        self.J = int(num_joints)
        self.device = torch.device(device)

        # History on CPU (stable)
        self.state_hist = torch.zeros((1, self.W, self.J, 3), device="cpu")
        self.target_hist = torch.zeros((1, self.W, self.J, 1), device="cpu")
        self.cmdvel_hist = torch.zeros((1, self.W, 3), device="cpu")
        self.hist_idx = 0

        # Predictions: desc_pred (1,J,4) for indices [6,7,9,10], pol_pred (1,1) for mass
        self.has_prediction = False
        self.desc_pred = torch.zeros((1, self.J, 4), device=self.device)
        self.pol_pred = torch.zeros((1, 1), device=self.device)

        self._step = 0
        self.model = self._build_model()

    def _build_model(self):
        import model_explicit
        return model_explicit.get_policy(self.W, self.device)

    def load_checkpoint(self, ckpt_path: str):
        adaptation_state = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(adaptation_state)
        except Exception:
            self.model.load_state_dict(adaptation_state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Adaptation checkpoint loaded from {ckpt_path}")

    def _forward_model(self, state_seq: torch.Tensor, target_seq: torch.Tensor, cmdvel_seq: torch.Tensor):
        with torch.no_grad():
            return self.model(state_seq, target_seq, cmdvel_seq)

    def reset(self):
        self.state_hist.zero_()
        self.target_hist.zero_()
        self.cmdvel_hist.zero_()
        self.hist_idx = 0
        self.has_prediction = False
        self._step = 0

    def add_history(self, state: torch.Tensor, action: torch.Tensor, cmd_vel: torch.Tensor):
        """
        Add NEW sample to history (for NEXT step prediction).

        state:  (1,J,3)  dynamic_joint_state
        action: (1,J) or (1,J,1)
        cmd_vel:(1,3)
        """
        state_cpu = state.detach().to("cpu")
        action_cpu = action.detach().to("cpu")
        cmd_cpu = cmd_vel.detach().to("cpu")

        if action_cpu.dim() == 2:
            action_cpu = action_cpu.unsqueeze(-1)  # (1,J,1)

        if self.hist_idx < self.W:
            self.state_hist[0, self.hist_idx] = state_cpu[0]
            self.target_hist[0, self.hist_idx] = action_cpu[0]
            self.cmdvel_hist[0, self.hist_idx] = cmd_cpu[0]
            self.hist_idx += 1
        else:
            # shift (simple & stable)
            self.state_hist[0, :-1] = self.state_hist[0, 1:].clone()
            self.target_hist[0, :-1] = self.target_hist[0, 1:].clone()
            self.cmdvel_hist[0, :-1] = self.cmdvel_hist[0, 1:].clone()
            self.state_hist[0, -1] = state_cpu[0]
            self.target_hist[0, -1] = action_cpu[0]
            self.cmdvel_hist[0, -1] = cmd_cpu[0]

    def update_predictions(self):
        """
        If history full and due, run model_explicit and update:
          desc_pred (1,J,4): [0]=joint_nominal, [1]=torque_limit, [2]=joint_range1, [3]=joint_range2
          pol_pred (1,1): mass
        """
        self._step += 1
        due = (self._step % self.adaptation_freq == 0)
        ready = (self.hist_idx >= self.W)

        if (not ready) or (not due):
            return

        if self.model is None:
            return

        state_seq = self.state_hist.to(self.device)
        target_seq = self.target_hist.to(self.device)
        cmdvel_seq = self.cmdvel_hist.to(self.device)

        desc_pred, pol_pred = self._forward_model(state_seq, target_seq, cmdvel_seq)

        self.desc_pred[...] = desc_pred
        self.pol_pred[...] = pol_pred
        self.has_prediction = True


# ---------------------------------------------------
# URMA + Adaptation runner for deployment (single env)
# ---------------------------------------------------
class DeploymentUrmaAdaptationRunner:
    """
    Keeps adaptation logic isolated from RobotHandler I/O.

    Per-step order:
      1) update_predictions (from past history)
      2) inject predictions
      3) compute action
      4) add_history (state, action, cmd_vel) for next prediction
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        metadata: SimpleNamespace,
        window_length: int = 10,
        adaptation_freq: int = 1,
        enable_adaptation: bool = True,
        verbose: bool = True,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval()
        self.metadata = metadata

        self.enable_adaptation = bool(enable_adaptation)
        self.verbose = bool(verbose)

        self.adaptation = None
        if self.enable_adaptation:
            self.adaptation = DeploymentAdaptationModule(
                window_length=window_length,
                adaptation_freq=adaptation_freq,
                num_joints=metadata.nr_dynamic_joint_observations,
                device=device,
            )

        self._printed_active = False
        self.global_step = 0
        self.adp_jr = "default"
        self.adp_mass = "default"

    def load_adaptation_checkpoint(self, ckpt_path: str):
        if self.adaptation is None:
            return
        self.adaptation.load_checkpoint(ckpt_path)

    def step(self, one_policy_observation: torch.Tensor, 
             correct_jr_flag: bool = False, 
             correct_joint_lower: torch.Tensor = None, 
             correct_joint_upper: torch.Tensor = None, 
             correct_mass_flag: bool = False, 
             correct_mass: torch.Tensor = None):
        """
        one_policy_observation: torch.float32, shape (obs_dim,)
        correct_jr_flag: bool, whether to use correct joint range
        correct_joint_lower: torch.float32, shape (12,), in policy order; hard lower limits for desc[:,9]
        correct_joint_upper: torch.float32, shape (12,), in policy order; hard upper limits for desc[:,10]
        correct_mass: normalized ground truth mass, ((mass_kg)/85)-1; used for policy and for logging gt_mass
        Returns:
          action: (12,) torch
          debug: dict (includes log values for DeploymentAdaptationLogger)
        """
        one_policy_observation = one_policy_observation.to(self.device)    

        # Parse observation (10-element general_policy_state: trunk+goal+gravity+mass)
        desc_gt, state, gps = one_policy_observation_to_inputs(one_policy_observation, self.metadata)

        # Get adaptation predictions (if enabled) - use existing predictions from previous step
        if self.enable_adaptation and (self.adaptation is not None):
            has_pred = self.adaptation.has_prediction
            desc_adapted = self.adaptation.desc_pred   # (1, J, 4) for indices [6, 7, 9, 10]
            pol_adapted = self.adaptation.pol_pred    # (1, 1) for mass
        else:
            has_pred = False
            desc_adapted = torch.zeros((1, self.metadata.nr_dynamic_joint_observations, 4), device=self.device)
            pol_adapted = torch.zeros((1, 1), device=self.device)

        # Decide what to feed policy
        desc_used = desc_gt.clone()
        gps_used = gps.clone()

        # Inject predictions if available and conditions met (matches eval_actor_urma_adaptive)
        cmd_vel = gps[3:6]  # goal velocities
        if self.enable_adaptation and (self.adaptation is not None) and has_pred and cmd_vel.norm() > CMD_VEL_THRESHOLD:
            # desc_pred[:,:,0]->index 6, [1]->7; joint range (9,10) use correct hard limits (policy order)
            desc_used[:, 6] = desc_adapted[0, :, 0]   # joint_nominal_position
            desc_used[:, 7] = desc_adapted[0, :, 1]   # torque_limit
            
            if correct_jr_flag and correct_joint_lower is not None and correct_joint_upper is not None:
                correct_joint_lower = torch.as_tensor(correct_joint_lower, dtype=torch.float32, device=self.device)
                correct_joint_upper = torch.as_tensor(correct_joint_upper, dtype=torch.float32, device=self.device)
                desc_used[:, 9] = correct_joint_lower
                desc_used[:, 10] = correct_joint_upper
                self.adp_jr = "correct"
            else:
                desc_used[:, 9] = desc_adapted[0, :, 2]
                desc_used[:, 10] = desc_adapted[0, :, 3]
                self.adp_jr = "adaptation"
            
            if correct_mass_flag and correct_mass is not None:
                correct_mass = torch.as_tensor(correct_mass, dtype=torch.float32, device=self.device)
                gps_used[..., -1] = correct_mass
                self.adp_mass = "correct"
            else:
                gps_used[..., -1] = pol_adapted[0, 0]     # mass
                self.adp_mass = "adaptation"
            
            if self.verbose and (not self._printed_active):
                print("[ADAPTATION] active (injecting desc[6,7,9,10] + mass)")
                self._printed_active = True
        else:
            self.adp_jr = "default"
            self.adp_mass = "default"

        # Run policy
        with torch.no_grad():
            action = self.policy(
                desc_used.unsqueeze(0),
                state.unsqueeze(0),
                gps_used.unsqueeze(0),
            ).squeeze(0)

        # Update adaptation with new action (matches simulation order)
        if self.enable_adaptation and (self.adaptation is not None):
            cmd_vel = gps[3:6]  # goal velocities
            self.adaptation.add_history(
                state=state.unsqueeze(0),
                action=action.unsqueeze(0),
                cmd_vel=cmd_vel.unsqueeze(0),
            )
            # Update predictions AFTER adding history (matches simulation)
            self.adaptation.update_predictions()

            if self.verbose and (not self.adaptation.has_prediction):
                print(f"[ADAPTATION] warming up: hist={self.adaptation.hist_idx}/{self.adaptation.W}")

        # Log values for DeploymentAdaptationLogger
        # Target shapes for visualize_predictions.py: (1, 12, 2) for joint range, (1, 1) for mass
        # Deployment: desc_gt/desc_used are (12, 18), desc_adapted is (1, 12, 4), gps is (10,)
        def _mass_denorm(x):
            return (torch.as_tensor(x, device=self.device) + 1) * 85

        def _to_log_jr(x):
            """Ensure (1, 12, 2) for joint range log."""
            t = torch.as_tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            return t

        def _to_log_mass(x):
            """Ensure (1, 1) for mass log."""
            t = torch.as_tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device)
            return t.reshape(1, 1) if t.dim() <= 1 else t

        # Ground truth joint range (1, 12, 2)
        if correct_joint_lower is not None and correct_joint_upper is not None:
            gt_jr = torch.stack([
                torch.as_tensor(correct_joint_lower, device=self.device),
                torch.as_tensor(correct_joint_upper, device=self.device),
            ], dim=-1)
        else:
            gt_jr = desc_gt[:, 9:11] if desc_gt.dim() == 2 else desc_gt[:, :, 9:11]
        gt_jr = _to_log_jr(gt_jr)

        # Adapted joint range (1, 12, 2)
        adapted_jr = desc_adapted[:, :, 2:4]

        # Used-in-policy joint range (1, 12, 2)
        used_jr = desc_used[:, 9:11] if desc_used.dim() == 2 else desc_used[:, :, 9:11]
        used_jr = _to_log_jr(used_jr)

        # Mass values (1, 1) each
        gt_mass = _to_log_mass(_mass_denorm(correct_mass)) if correct_mass is not None else torch.tensor([[85.0]], device=self.device)
        adapted_mass = _to_log_mass(_mass_denorm(pol_adapted[0, 0]))
        used_mass = _to_log_mass(_mass_denorm(gps_used[..., -1]))

        debug = {
            "step": self.global_step,
            "command_velocity": cmd_vel.cpu().numpy().tolist(),
            "has_prediction": has_pred,
            "hist_idx": (self.adaptation.hist_idx if self.adaptation else None),
            "adp_jr": self.adp_jr,
            "adp_mass": self.adp_mass,
            "log": {
                "ground_truth_joint_range": gt_jr,
                "adapted_joint_range": adapted_jr,
                "used_in_policy_joint_range": used_jr,
                "ground_truth_mass": gt_mass,
                "adapted_mass": adapted_mass,
                "used_in_policy_mass": used_mass,
                "has_adaptation": has_pred,
            },
        }
        self.global_step += 1
        return action, debug


# -------------------------
# Deployment adaptation logger (JSON format matching eval_actor_urma_adaptive)
# -------------------------
class DeploymentAdaptationLogger:
    """Logs ground truth, adapted, and used-in-policy values for visualize_predictions.py."""

    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        self.log = {
            "timesteps": [],
            "ground_truth_joint_range": [],
            "adapted_joint_range": [],
            "used_in_policy_joint_range": [],
            "ground_truth_mass": [],
            "adapted_mass": [],
            "used_in_policy_mass": [],
            "has_adaptation": [],
        }

    def reset(self):
        """Start fresh log (call when entering nn mode)."""
        self.log = {
            "timesteps": [],
            "ground_truth_joint_range": [],
            "adapted_joint_range": [],
            "used_in_policy_joint_range": [],
            "ground_truth_mass": [],
            "adapted_mass": [],
            "used_in_policy_mass": [],
            "has_adaptation": [],
        }

    def log_timestep(
        self,
        timestep: int,
        ground_truth_joint_range,
        adapted_joint_range,
        used_in_policy_joint_range,
        ground_truth_mass,
        adapted_mass,
        used_in_policy_mass,
        has_adaptation: bool,
    ):
        """Append one timestep. All tensors/arrays in policy order, shapes (1,12,2) or (1,1) for mass."""
        self.log["timesteps"].append(timestep)
        self.log["ground_truth_joint_range"].append(
            np.asarray(ground_truth_joint_range).tolist()
        )
        self.log["adapted_joint_range"].append(
            np.asarray(adapted_joint_range).tolist()
        )
        self.log["used_in_policy_joint_range"].append(
            np.asarray(used_in_policy_joint_range).tolist()
        )
        self.log["ground_truth_mass"].append(
            np.asarray(ground_truth_mass).tolist()
        )
        self.log["adapted_mass"].append(np.asarray(adapted_mass).tolist())
        self.log["used_in_policy_mass"].append(
            np.asarray(used_in_policy_mass).tolist()
        )
        # Shape (num_envs,) for visualize_predictions: [[True], [True], ...] -> (T, 1)
        self.log["has_adaptation"].append([has_adaptation])

    def save_to_json(self, filepath: str):
        """Save to JSON (compatible with visualize_predictions.py)."""
        data = {
            **self.log,
            "num_envs": 1,
            "num_joints": self.num_joints,
            "description": {
                "ground_truth": "Real physical properties (hard_joint_limits)",
                "adapted": "Predictions from adaptation module",
                "used_in_policy": "Values in policy observation",
                "has_adaptation": "True if robot has valid prediction",
            },
        }
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Logged {len(self.log['timesteps'])} timesteps to {filepath}")


# -------------------------
# Joint position logger (for visualize_joint_position.py)
# -------------------------
class JointPositionLogger:
    """Logs joint positions per timestep (policy order)."""

    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        self.log = {"timesteps": [], "joint_position": []}

    def reset(self):
        """Start fresh log (call when entering nn mode)."""
        self.log = {"timesteps": [], "joint_position": []}

    def log_timestep(self, timestep: int, joint_position):
        """Append one timestep. joint_position: (12,) in policy order."""
        self.log["timesteps"].append(timestep)
        self.log["joint_position"].append(np.asarray(joint_position).tolist())

    def save_to_json(self, filepath: str):
        """Save to JSON."""
        data = {
            **self.log,
            "num_joints": self.num_joints,
            "description": "Joint positions (rad) in policy order: hip(fl,fr,rl,rr), thigh(fl,fr,rl,rr), knee(fl,fr,rl,rr)",
        }
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Logged {len(self.log['timesteps'])} timesteps to {filepath}")


# -------------------------
# RobotHandler (deployment)
# -------------------------
class RobotHandler(Node):
    def __init__(self):
        super().__init__("robot_handler")

        # --- Unitree channel & state client ---
        ChannelFactoryInitialize(0, "eno1")
        rsc = RobotStateClient()
        rsc.Init()
        rsc.ServiceSwitch("sport_mode", False)

        # --- ROS QoS ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.low_state_subscription = self.create_subscription(
            LowState,
            "/lowstate",
            self.low_state_callback,
            qos_profile=qos_profile
        )

        self.joystick_subscription = self.create_subscription(
            WirelessController,
            "/wirelesscontroller",
            self.joystick_callback,
            qos_profile=qos_profile
        )

        self.publisher = self.create_publisher(
            LowCmd,
            "/lowcmd",
            qos_profile=qos_profile
        )

        # --- Default LowCmd (UNCHANGED) ---
        self.default_low_cmd = LowCmd()
        self.default_low_cmd.head[0] = 0xFE
        self.default_low_cmd.head[1] = 0xEF
        self.default_low_cmd.level_flag = 0xFF
        for i in range(20):
            self.default_low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.default_low_cmd.motor_cmd[i].q = 0.0
            self.default_low_cmd.motor_cmd[i].dq = 0.0
            self.default_low_cmd.motor_cmd[i].kp = 0.0
            self.default_low_cmd.motor_cmd[i].kd = 0.0
            self.default_low_cmd.motor_cmd[i].tau = 0.0

        # --- Robot state (UNCHANGED) ---
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)

        self.control_frequency = 50.0

        # Policy-side nominal in robot order (fr,fl,rr,rl)
        self.nominal_joint_positions = np.array([
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 1.0, -1.5,
            0.1, 1.0, -1.5
        ])

        self.lying_joint_positions = np.array([
            -0.04584759, 1.26458573, -2.79743123,
            0.03388786, 1.25516927, -2.7853148,
            -0.34251189, 1.27808392, -2.8028338,
            0.34323859, 1.27829576, -2.81149054
        ])

        self.stand_and_lie_seconds = 1.0
        self.stand_and_lie_p_gain = 70.0
        self.stand_and_lie_d_gain = 3.0

        self.velocity_safety_threshold = 15.0
        self.goal_clip = 0.5

        self.nn_p_gain = 20.0
        self.nn_d_gain = 0.5
        self.scaling_factor = 0.3
        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        # robot side order: fr(hip, thigh, knee), fl(hip, thigh, knee), rr(hip, thigh, knee), rl(hip, thigh, knee).
        # policy side order: hip(fl, fr, rl, rr), thigh(fl, fr, rl, rr), knee(fl, fr, rl, rr)
        self.obs_reorder_mask = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.action_reorder_mask = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

        self.previous_control_mode = None
        self.last_seen_control_mode = None
        self.control_mode = None

        # --- On-the-fly locking: default vs locking condition ---
        self.locking_active = False  # Always False when entering nn; button 2 toggles to True
        self.joint_ids_to_lock = JOINT_IDS_TO_LOCK
        self.hard_locked_factor = HARD_LOCKED_FACTOR
        self.joint_limits = np.array([
            [-1.05,  1.05 ],
            [-1.57,  3.49 ],
            [-2.72, -0.84 ],
            [-1.05,  1.05 ],
            [-1.57,  3.49 ],
            [-2.72, -0.84 ],
            [-1.05,  1.05 ],
            [-0.52,  4.53 ],
            [-2.72, -0.84 ],
            [-1.05,  1.05 ],
            [-0.52,  4.53 ],
            [-2.72, -0.84 ],
        ])

        # Default: full joint range (no locking)
        self.default_hard_joint_lower_limits = self.joint_limits[:, 0].copy()
        self.default_hard_joint_upper_limits = self.joint_limits[:, 1].copy()

        # Locking: narrowed range for JOINT_IDS_TO_LOCK
        self.locked_hard_joint_lower_limits = self.joint_limits[:, 0].copy()
        self.locked_hard_joint_lower_limits[self.joint_ids_to_lock] = (
            self.hard_locked_factor * self.joint_limits[self.joint_ids_to_lock, 0]
            + (1 - self.hard_locked_factor) * self.nominal_joint_positions[self.joint_ids_to_lock]
        )
        self.locked_hard_joint_upper_limits = self.joint_limits[:, 1].copy()
        self.locked_hard_joint_upper_limits[self.joint_ids_to_lock] = (
            self.hard_locked_factor * self.joint_limits[self.joint_ids_to_lock, 1]
            + (1 - self.hard_locked_factor) * self.nominal_joint_positions[self.joint_ids_to_lock]
        )

        # Active limits (selected by locking_active)
        self.hard_joint_lower_limits = self.default_hard_joint_lower_limits.copy()
        self.hard_joint_upper_limits = self.default_hard_joint_upper_limits.copy()

        self.high_p_gain = np.ones(12) * 60.0
        self.high_d_gain = np.ones(12) * 1.0

        # --- Policy load (UNCHANGED) ---
        self.load_policy("2026-02-15_17-56-39/model_11000.pt")

        # --- Online adaptation runner (CPU, freq=1). Model placeholders inside. ---
        self.urma_runner = DeploymentUrmaAdaptationRunner(
            policy=self.policy,
            metadata=self.metadata,
            window_length=10,
            adaptation_freq=1,
            enable_adaptation=ENABLE_ADAPTATION,
            verbose=True,
            device="cpu",
        )
        # Use model_explicit checkpoint (from online_urma_explicit_offline_adaptation)
        self.urma_runner.load_adaptation_checkpoint("adaptation_module/2026-02-21_17-01-32_online_urma_explicit_offline_adaptation_GenBot1K-adp-v3_training/explicit_v2_model_epoch85.pth")

        # --- Adaptation logger (start fresh on nn enter, save on nn exit) ---
        self.adaptation_logger = DeploymentAdaptationLogger(num_joints=12)
        self.joint_position_logger = JointPositionLogger(num_joints=12)
        self.adaptation_log_dir = os.path.join(os.path.dirname(__file__), "adaptation_logs")
        self.joint_position_log_dir = os.path.join(os.path.dirname(__file__), "joint_position_logs")

        print("Robot ready. Expert policy + online adaptation runner initialized (CPU).")

        self.timer = self.create_timer(1 / self.control_frequency, self.timer_callback)

    # ----------------
    # ROS callbacks
    # ----------------
    def switch_control_mode(self, control_mode):
        prev = self.control_mode
        self.previous_control_mode = prev if self.control_mode != control_mode else self.previous_control_mode
        self.control_mode = control_mode

        # Entering nn: start fresh log, always default (no locking)
        if control_mode == "nn":
            self.locking_active = False
            self._apply_active_hard_limits()
            self.adaptation_logger.reset()
            self.joint_position_logger.reset()
            print("[LOG] Started fresh adaptation log (nn mode entered)")
        # Exiting nn to stand_up/lie_down/stop: save logs (same timestamp, in timestamp subfolders)
        elif prev == "nn" and control_mode in ("stand_up", "lie_down", "stop"):
            if len(self.adaptation_logger.log["timesteps"]) > 0:
                stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                adaptation_stamp_dir = os.path.join(self.adaptation_log_dir, stamp)
                joint_position_stamp_dir = os.path.join(self.joint_position_log_dir, stamp)
                self.adaptation_logger.save_to_json(
                    os.path.join(adaptation_stamp_dir, "adaptation_log.json")
                )
                self.joint_position_logger.save_to_json(
                    os.path.join(joint_position_stamp_dir, "joint_position_log.json")
                )
            else:
                print("[LOG] No timesteps logged, skipping save")

        print(f"Switched to control mode: {self.control_mode}")

    def _apply_active_hard_limits(self):
        """Update hard_joint_lower/upper_limits from locking_active."""
        if self.locking_active:
            self.hard_joint_lower_limits[:] = self.locked_hard_joint_lower_limits
            self.hard_joint_upper_limits[:] = self.locked_hard_joint_upper_limits
        else:
            self.hard_joint_lower_limits[:] = self.default_hard_joint_lower_limits
            self.hard_joint_upper_limits[:] = self.default_hard_joint_upper_limits

    def low_state_callback(self, msg):
        motor_states = msg.motor_state
        self.joint_positions = np.array([motor_states[i].q for i in range(12)])
        self.joint_velocities = np.array([motor_states[i].dq for i in range(12)])
        self.orientation = np.array([
            msg.imu_state.quaternion[1],
            msg.imu_state.quaternion[2],
            msg.imu_state.quaternion[3],
            msg.imu_state.quaternion[0],
        ])
        self.angular_velocity = msg.imu_state.gyroscope

    def joystick_callback(self, msg):
        self.x_goal_velocity = np.clip(msg.ly, -self.goal_clip, self.goal_clip)
        self.y_goal_velocity = np.clip(-msg.lx, -self.goal_clip, self.goal_clip)
        self.yaw_goal_velocity = np.clip(-msg.rx, -self.goal_clip, self.goal_clip)

        if msg.keys == 2048:
            self.switch_control_mode("stand_up")
        elif msg.keys == 1024:
            self.switch_control_mode("lie_down")
        elif msg.keys == 512:
            self.switch_control_mode("nn")
        elif msg.keys == 256:
            self.switch_control_mode("stop")
        # On-the-fly locking: 1=default, 2=locking (no-op if already in that state)
        elif msg.keys == 1:
            if self.locking_active:
                self.locking_active = False
                self._apply_active_hard_limits()
                print("[LOCKING] Switched to default (full joint range)")
        elif msg.keys == 2:
            if not self.locking_active:
                self.locking_active = True
                self._apply_active_hard_limits()
                print("[LOCKING] Switched to locking condition")

    # ----------------
    # Main loop
    # ----------------
    def timer_callback(self):
        # KEEP ORIGINAL safety logic (no abs)
        if np.max(self.joint_velocities) > self.velocity_safety_threshold:
            print("Velocity safety threshold exceeded.")
            self.switch_control_mode("stand_up")

        if self.control_mode == "stand_up":
            self.stand_up()
        elif self.control_mode == "lie_down":
            self.lie_down()
        elif self.control_mode == "nn":
            self.nn()
        elif self.control_mode == "stop":
            ...

        self.last_seen_control_mode = self.control_mode

    # ----------------
    # Stand/Lie (UNCHANGED)
    # ----------------
    def stand_up(self):
        if self.last_seen_control_mode != "stand_up":
            self.standing_delta = self.nominal_joint_positions - self.joint_positions
            self.standing_intermediate_position = deepcopy(self.joint_positions)
            self.standing_counter = 0

        if self.standing_counter < self.stand_and_lie_seconds * self.control_frequency:
            self.standing_counter += 1
            self.standing_intermediate_position += self.standing_delta / (self.stand_and_lie_seconds * self.control_frequency)
            target_positions = self.standing_intermediate_position
        else:
            target_positions = self.nominal_joint_positions

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(12):
            low_cmd.motor_cmd[i].q = target_positions[i]
            low_cmd.motor_cmd[i].kp = self.stand_and_lie_p_gain
            low_cmd.motor_cmd[i].kd = self.stand_and_lie_d_gain

        low_cmd.crc = get_crc(low_cmd)
        self.publisher.publish(low_cmd)

    def lie_down(self):
        if self.last_seen_control_mode != "lie_down":
            self.lying_delta = self.lying_joint_positions - self.joint_positions
            self.lying_intermediate_position = deepcopy(self.joint_positions)
            self.lying_counter = 0

        if self.lying_counter < self.stand_and_lie_seconds * self.control_frequency:
            self.lying_counter += 1
            self.lying_intermediate_position += self.lying_delta / (self.stand_and_lie_seconds * self.control_frequency)
            target_positions = self.lying_intermediate_position
        else:
            target_positions = self.lying_joint_positions

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(12):
            low_cmd.motor_cmd[i].q = target_positions[i]
            low_cmd.motor_cmd[i].kp = self.stand_and_lie_p_gain
            low_cmd.motor_cmd[i].kd = self.stand_and_lie_d_gain

        low_cmd.crc = get_crc(low_cmd)
        self.publisher.publish(low_cmd)

    # ----------------
    # Observation construction (UNCHANGED)
    # ----------------
    def construct_go2_observation(self, qpos, qvel, previous_action, trunk_angular_velocity, goal_velocities, projected_gravity_vector):
        """
        EXACTLY your original deployment observation builder.
        (Keeps all fields + normalization + ordering you already validated.)
        """
        self.relative_joint_position_normalized = np.array([
            [0.9258, 0.6198, 0.8478],
            [0.9258, 0.3802, 0.8478],
            [0.3186, 0.6198, 0.8478],
            [0.3186, 0.3802, 0.8478],
            [0.9258, 0.8634, 0.8726],
            [0.9258, 0.1366, 0.8726],
            [0.3186, 0.8634, 0.8726],
            [0.3186, 0.1366, 0.8726],
            [0.6792, 0.9014, 0.4876],
            [0.6792, 0.0986, 0.4876],
            [0.0294, 0.8928, 0.5740],
            [0.0294, 0.1072, 0.5740]
        ])
        self.relative_joint_axis_local = np.array([
            [ 1.0000e+00,  2.1027e-09,  1.0610e-09],
            [ 1.0000e+00, -1.8324e-09,  6.3952e-10],
            [ 1.0000e+00,  4.6547e-11, -9.3016e-10],
            [ 1.0000e+00, -1.0698e-09, -2.7439e-09],
            [-2.3842e-07,  9.9500e-01,  9.9833e-02],
            [ 0.0000e+00,  9.9500e-01, -9.9833e-02],
            [-2.3842e-07,  9.9500e-01,  9.9833e-02],
            [ 1.7881e-07,  9.9500e-01, -9.9833e-02],
            [-1.1921e-07,  9.9500e-01,  9.9833e-02],
            [ 0.0000e+00,  9.9500e-01, -9.9833e-02],
            [-1.1921e-07,  9.9500e-01,  9.9834e-02],
            [ 0.0000e+00,  9.9500e-01, -9.9833e-02]
        ])
        self.joint_nominal_positions = np.array([ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,
                                                 1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000])
        self.desc_joint_max_torque = np.array([23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000,
                                               23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000])
        self.joint_max_velocity = np.array([30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.])
        self.desc_joint_lower_limits = np.array([-1.0500, -1.0500, -1.0500, -1.0500, -1.5700, -1.5700,
                                                 -0.5200, -0.5200, -2.7200, -2.7200, -2.7200, -2.7200])
        self.desc_joint_upper_limits = np.array([ 1.0500,  1.0500,  1.0500,  1.0500,  3.4900,  3.4900,
                                                  4.5300,  4.5300, -0.8400, -0.8400, -0.8400, -0.8400])

        self.gains_and_action_scaling_factor = np.array([20.0000,  0.5000,  0.3000])
        self.desc_mass = np.array([15.017])
        self.robot_dimensions = np.array([0.6196, 0.3902, 0.3835])
        self.nr_joints = np.array([12])
        self.scaled_foot_size = np.array([-0.8240, -0.8240, -0.8240])

        dynamic_joint_observations = np.empty(0)
        for i in range(12):
            desc_vector = np.concatenate([
                (self.relative_joint_position_normalized[i, :] / 0.5) - 1.0,
                self.relative_joint_axis_local[i, :],
                np.array([self.joint_nominal_positions[i]]) / 4.6,
                (np.array([self.desc_joint_max_torque[i]]) / 500.0) - 1.0,
                (np.array([self.joint_max_velocity[i]]) / 17.5) - 1.0,
                np.array([self.desc_joint_lower_limits[i]]),
                np.array([self.desc_joint_upper_limits[i]]),
                ((self.gains_and_action_scaling_factor / np.array([50.0, 1.0, 0.4])) - 1.0),
                ((self.desc_mass / 85.0) - 1.0) * 0,
                ((self.robot_dimensions / 1.0) - 1.0) * 0,
            ])

            current_observation = np.concatenate((
                desc_vector,
                np.array([qpos[i]]),
                np.array([qvel[i]]),
                np.array([previous_action[i]])
            ))
            dynamic_joint_observations = np.concatenate((dynamic_joint_observations, current_observation))

        observation = np.concatenate([
            dynamic_joint_observations,
            np.clip(trunk_angular_velocity / 50.0, -10.0, 10.0),
            goal_velocities,
            projected_gravity_vector,
            (self.desc_mass / 85.0) - 1.0,
        ])

        return observation

    # ----------------
    # NN control (hardware logic kept; only swapped policy call path)
    # ----------------
    def nn(self):
        # KEEP ORIGINAL gating
        if self.last_seen_control_mode != "nn" and self.last_seen_control_mode != "stand_up":
            return

        if self.last_seen_control_mode != "nn":
            self.previous_action = np.zeros(12)

        # dynamic joint state (policy order) - KEEP ORIGINAL
        qpos = (self.joint_positions - self.nominal_joint_positions) / 4.6
        qpos = qpos[self.obs_reorder_mask]

        qvel = self.joint_velocities / 35.0
        qvel = qvel[self.obs_reorder_mask]

        previous_action = self.previous_action / 10.0
        previous_action = previous_action[self.obs_reorder_mask]

        # general policy state variant - KEEP ORIGINAL
        trunk_angular_velocity = self.angular_velocity
        goal_velocities = np.array([self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity])
        orientation_quat_inv = R.from_quat(self.orientation).inv()
        projected_gravity_vector = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

        # Build observation (KEEP ORIGINAL)
        observation = self.construct_go2_observation(
            qpos, qvel, previous_action,
            trunk_angular_velocity, goal_velocities, projected_gravity_vector
        )
        observation = torch.from_numpy(observation).float()

        # Correct joint range (hard limits) in policy order: policy p -> robot obs_reorder_mask[p]
        # When locking_active: use locked limits; otherwise use default limits
        correct_joint_lower = self.hard_joint_lower_limits[np.array(self.obs_reorder_mask)]
        correct_joint_upper = self.hard_joint_upper_limits[np.array(self.obs_reorder_mask)]
        correct_mass = ((self.desc_mass + ADDED_MASS) / 85.0) - 1.0

        # Runner: always use correct joint range (physical limits for current mode)
        action_t, debug = self.urma_runner.step(
            observation,
            correct_jr_flag=True,
            correct_joint_lower=correct_joint_lower,
            correct_joint_upper=correct_joint_upper,
            correct_mass_flag=CORRECT_MASS_FLAG,
            correct_mass=correct_mass,
        )

        # Log joint positions (policy order)
        joint_pos_policy_order = self.joint_positions[np.array(self.obs_reorder_mask)]
        self.joint_position_logger.log_timestep(
            timestep=debug["step"],
            joint_position=joint_pos_policy_order,
        )

        # Log adaptation data (for visualize_predictions.py)
        log_data = debug.get("log")
        if log_data:
            self.adaptation_logger.log_timestep(
                timestep=debug["step"],
                ground_truth_joint_range=log_data["ground_truth_joint_range"].cpu().numpy(),
                adapted_joint_range=log_data["adapted_joint_range"].cpu().numpy(),
                used_in_policy_joint_range=log_data["used_in_policy_joint_range"].cpu().numpy(),
                ground_truth_mass=log_data["ground_truth_mass"].cpu().numpy(),
                adapted_mass=log_data["adapted_mass"].cpu().numpy(),
                used_in_policy_mass=log_data["used_in_policy_mass"].cpu().numpy(),
                has_adaptation=log_data["has_adaptation"],
            )

        # KEEP ORIGINAL downstream logic: reorder + convert to numpy
        action = action_t.detach().cpu().numpy()
        action = action[self.action_reorder_mask]

        target_joint_positions = self.nominal_joint_positions + self.scaling_factor * action

        # KEEP ORIGINAL hard-locking / gain logic (even if p_gains/d_gains not used later)
        hard_lower_diff = -(self.joint_positions[self.joint_ids_to_lock] - self.hard_joint_lower_limits[self.joint_ids_to_lock])
        hard_upper_diff = self.joint_positions[self.joint_ids_to_lock] - self.hard_joint_upper_limits[self.joint_ids_to_lock]

        hard_out_of_lower_limits = hard_lower_diff > 0
        hard_out_of_upper_limits = hard_upper_diff > 0
        hard_out_of_limits = hard_out_of_lower_limits | hard_out_of_upper_limits

        p_gains = np.ones(12) * self.nn_p_gain
        d_gains = np.ones(12) * self.nn_d_gain

        p_gains[self.joint_ids_to_lock] = np.where(
            hard_out_of_limits, self.high_p_gain[self.joint_ids_to_lock], p_gains[self.joint_ids_to_lock]
        )
        d_gains[self.joint_ids_to_lock] = np.where(
            hard_out_of_limits, self.high_d_gain[self.joint_ids_to_lock], d_gains[self.joint_ids_to_lock]
        )

        target_joint_positions[self.joint_ids_to_lock] = np.clip(
            target_joint_positions[self.joint_ids_to_lock],
            self.hard_joint_lower_limits[self.joint_ids_to_lock],
            self.hard_joint_upper_limits[self.joint_ids_to_lock]
        )

        # KEEP ORIGINAL command sending (kp/kd set to nn gains globally)
        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(12):
            low_cmd.motor_cmd[i].q = target_joint_positions[i]
            low_cmd.motor_cmd[i].kp = self.nn_p_gain
            low_cmd.motor_cmd[i].kd = self.nn_d_gain

        low_cmd.crc = get_crc(low_cmd)

        # KEEP ORIGINAL dt debug + add adaptation flags
        now = time.time()
        if hasattr(self, "last_publish_time_wall"):
            cv = debug.get("command_velocity") or [0, 0, 0]
            cv_str = ", ".join(f"{v:.2f}" for v in cv[:3])
            print("dt:", f"{now - self.last_publish_time_wall:.3f}", 
                  "| command velocity", f"[{cv_str}]",
                  "| hist:", debug.get("hist_idx"), 
                  "| adp_jr:", debug.get("adp_jr"), 
                  "| adp_mass:", debug.get("adp_mass")
                  )
        self.last_publish_time_wall = now

        self.publisher.publish(low_cmd)

        # KEEP ORIGINAL
        self.previous_action = action

    # ----------------
    # Policy loading (UNCHANGED)
    # ----------------
    def load_policy(self, model_file_name):
        current_path = os.path.dirname(__file__)
        checkpoint_dir = os.path.join(current_path, "policies")
        model_path = os.path.join(checkpoint_dir, model_file_name)

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        actor_state_dict = {
            k.replace("actor.", "", 1): v
            for k, v in state_dict["model_state_dict"].items()
            if k.startswith("actor.")
        }

        self.policy = get_policy(torch.device("cpu"))
        self.policy.load_state_dict(actor_state_dict)
        self.policy.eval()

        self.metadata_dict = {
            "nr_dynamic_joint_observations": 12,
            "single_dynamic_joint_observation_length": 21,
            "dynamic_joint_observation_length": 252,
            "dynamic_joint_description_size": 18,
            "trunk_angular_vel_update_obs_idx": [252, 253, 254],
            "goal_velocity_update_obs_idx": [255, 256, 257],
            "projected_gravity_update_obs_idx": [258, 259, 260],
        }
        self.metadata = SimpleNamespace(**self.metadata_dict)


def main(args=None):
    rclpy.init(args=args)
    robot_handler = RobotHandler()
    rclpy.spin(robot_handler)
    robot_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
