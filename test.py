import os
import shutil
import json
import onnx
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from unitree_go.msg import LowState, LowCmd, WirelessController

from crc_module import get_crc

from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

import torch
from types import SimpleNamespace

from policy import get_policy

def one_policy_observation_to_inputs(one_policy_observation, metadata):
    """
        Transform one policy observation into 5 inputs that a one policy model accept
        Args:
            one_policy_observation (tensor): The one policy observation. eg. For GenDog1, the size is 340 on the last dimension.
            meta: could be anything that provide the metadata of robot joint and foot numbers. By default, pass in env.unwrapped.
            device: 
        """
    # Dynamic Joint Observations
    nr_dynamic_joint_observations = metadata.nr_dynamic_joint_observations
    single_dynamic_joint_observation_length = metadata.single_dynamic_joint_observation_length
    dynamic_joint_observation_length = metadata.dynamic_joint_observation_length
    dynamic_joint_description_size = metadata.dynamic_joint_description_size
    dynamic_joint_combined_state = one_policy_observation[..., :dynamic_joint_observation_length].view((*one_policy_observation.shape[:-1], nr_dynamic_joint_observations, single_dynamic_joint_observation_length))
    dynamic_joint_description = dynamic_joint_combined_state[..., :dynamic_joint_description_size]
    dynamic_joint_state = dynamic_joint_combined_state[..., dynamic_joint_description_size:]

    trunk_angular_vel_update_obs_idx = metadata.trunk_angular_vel_update_obs_idx
    goal_velocity_update_obs_idx = metadata.goal_velocity_update_obs_idx
    projected_gravity_update_obs_idx = metadata.projected_gravity_update_obs_idx
    general_policy_state = one_policy_observation[..., trunk_angular_vel_update_obs_idx+goal_velocity_update_obs_idx+projected_gravity_update_obs_idx]
    GENERAL_POLICY_STATE_LEN = 11
    general_policy_state = torch.cat((general_policy_state, one_policy_observation[..., -GENERAL_POLICY_STATE_LEN:]), dim=-1) # gains_and_action_scaling_factor; mass; robot_dimensions

    inputs = (
        dynamic_joint_description,
        dynamic_joint_state,
        general_policy_state
    )
    return inputs

class RobotHandler(Node):
    def __init__(self):
        super().__init__("robot_handler")

        ChannelFactoryInitialize(0, "eno1")
        rsc = RobotStateClient()
        rsc.Init()
        rsc.ServiceSwitch("sport_mode", False)

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

        self.default_low_cmd = LowCmd()
        self.default_low_cmd.head[0] = 0xFE
        self.default_low_cmd.head[1] = 0xEF
        self.default_low_cmd.level_flag = 0xFF
        for i in range(20):
            self.default_low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.default_low_cmd.motor_cmd[i].q = 0.0 # 2.146e9
            self.default_low_cmd.motor_cmd[i].dq = 0.0 # 16000.0
            self.default_low_cmd.motor_cmd[i].kp = 0.0
            self.default_low_cmd.motor_cmd[i].kd = 0.0
            self.default_low_cmd.motor_cmd[i].tau = 0.0

        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)

        self.control_frequency = 50.0

        self.nominal_joint_positions = np.array([
            0.0, 0.8, -1.5,
            0.0, 0.8, -1.5,
            0.0, 1.0, -1.5,
            0.0, 1.0, -1.5
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

        self.velocity_safety_threshold = 10.0
        self.goal_clip = 0.5

        self.nn_p_gain = 20.0
        self.nn_d_gain = 0.5
        self.scaling_factor = 0.3
        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        self.obs_reorder_mask = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.action_reorder_mask = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

        self.previous_control_mode = None
        self.last_seen_control_mode = None
        self.control_mode = None

        self.joint_ids_to_lock = [] # Put the joint ids that you want to lock here
        self.hard_locked_factor = 0.2
        self.joint_limits = np.array([
            [-1.0472,   1.0472 ],
            [-1.5708,   3.4907 ],
            [-2.7227,  -0.83776],
            [-1.0472,   1.0472 ],
            [-1.5708,   3.4907 ],
            [-2.7227,  -0.83776],
            [-1.0472,   1.0472 ],
            [-0.5236,   4.5379 ],
            [-2.7227,  -0.83776],
            [-1.0472,   1.0472 ],
            [-0.5236,   4.5379 ],
            [-2.7227,  -0.83776],
        ])
        self.hard_joint_lower_limits = self.joint_limits[:, 0]
        self.hard_joint_lower_limits[self.joint_ids_to_lock] = self.hard_locked_factor * self.hard_joint_lower_limits[self.joint_ids_to_lock] + (1 - self.hard_locked_factor) * self.nominal_joint_positions[self.joint_ids_to_lock]
        self.hard_joint_upper_limits = self.joint_limits[:, 1]
        self.hard_joint_upper_limits[self.joint_ids_to_lock] = self.hard_locked_factor * self.hard_joint_upper_limits[self.joint_ids_to_lock] + (1 - self.hard_locked_factor) * self.nominal_joint_positions[self.joint_ids_to_lock]
        self.high_p_gain = np.ones(12) * 60.0
        self.high_d_gain = np.ones(12) * 1.0

        self.load_policy("2025-11-23_23-56-25_model_5999.pt")

        print(f"Robot ready. Model running on {jax.default_backend()}.")

        self.timer = self.create_timer(1 / self.control_frequency, self.timer_callback)


    def switch_control_mode(self, control_mode):
        self.previous_control_mode = self.control_mode if self.control_mode != control_mode else self.previous_control_mode
        self.control_mode = control_mode
        print(f"Switched to control mode: {self.control_mode}")
    

    def low_state_callback(self, msg):
        motor_states = msg.motor_state
        self.joint_positions = np.array([motor_states[i].q for i in range(12)])

        self.joint_velocities = np.array([motor_states[i].dq for i in range(12)])

        self.orientation = np.array([msg.imu_state.quaternion[1], msg.imu_state.quaternion[2], msg.imu_state.quaternion[3], msg.imu_state.quaternion[0]])
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
    

    def timer_callback(self):
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

    def construct_go2_observation(self, qpos, qvel, previous_action, trunk_angular_velocity, goal_velocities, projected_gravity_vector):
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
                    (self.desc_mass / 85.0) - 1.0,
                    (self.robot_dimensions / 1.0) - 1.0,
                    ])
                    
                    current_observation = np.concatenate((desc_vector, np.array([qpos[i]]), np.array([qvel[i]]), np.array([previous_action[i]])))
                    dynamic_joint_observations = np.concatenate((dynamic_joint_observations, current_observation))
                
                observation = np.concatenate([
                    dynamic_joint_observations,
                    np.clip(trunk_angular_velocity / 50.0, -10.0, 10.0),
                    goal_velocities,
                    projected_gravity_vector,
                    ((self.gains_and_action_scaling_factor / np.array([50.0, 1.0, 0.4])) - 1.0),
                    (self.desc_mass / 85.0) - 1.0,
                    ((self.robot_dimensions / 1.0) - 1.0),
                    (self.nr_joints / 15.0 - 1.0),
                    self.scaled_foot_size
                ])
                
                return observation
        
    def nn(self):
        # Only run neural network if it's already running or if the robot is standing up
        if self.last_seen_control_mode != "nn" and self.last_seen_control_mode != "stand_up":
            return
        
        if self.last_seen_control_mode != "nn":
            self.previous_action = np.zeros(12)
        
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
                                                    [0.0294, 0.1072, 0.5740]])
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
                                                    [ 0.0000e+00,  9.9500e-01, -9.9833e-02]])
        self.joint_nominal_positions = np.array([ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000])
        self.desc_joint_max_torque = np.array([23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000, 23.5000])
        self.joint_max_velocity = np.array([30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.])
        self.desc_joint_lower_limits = np.array([-1.0500, -1.0500, -1.0500, -1.0500, -1.5700, -1.5700, -0.5200, -0.5200, -2.7200, -2.7200, -2.7200, -2.7200])
        self.desc_joint_upper_limits = np.array([ 1.0500,  1.0500,  1.0500,  1.0500,  3.4900,  3.4900,  4.5300,  4.5300, -0.8400, -0.8400, -0.8400, -0.8400])

        qpos = (self.joint_positions - self.nominal_joint_positions) / 4.6
        qpos = qpos[self.obs_reorder_mask]

        qvel = self.joint_velocities / 35.0
        qvel = qvel[self.obs_reorder_mask]

        previous_action = self.previous_action / 10.0
        previous_action = previous_action[self.obs_reorder_mask]
        
        # 9 elements
        trunk_angular_velocity = self.angular_velocity
        goal_velocities = np.array([self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity])
        orientation_quat_inv = R.from_quat(self.orientation).inv()
        projected_gravity_vector = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0]))

        # 11 elements
        self.gains_and_action_scaling_factor = np.array([20.0000,  0.5000,  0.3000])
        self.desc_mass = np.array([15.017])
        self.robot_dimensions = np.array([0.6196, 0.3902, 0.3835])
        self.nr_joints = np.array([12])
        self.scaled_foot_size = np.array([-0.8240, -0.8240, -0.8240])
        
        observation = self.construct_go2_observation(qpos, qvel, previous_action, trunk_angular_velocity, goal_velocities, projected_gravity_vector)
        observation = torch.from_numpy(observation).float()
        dynamic_joint_description, dynamic_joint_state, general_state = one_policy_observation_to_inputs(observation, self.metadata)
        action = self.policy(dynamic_joint_description.unsqueeze(0), dynamic_joint_state.unsqueeze(0), general_state.unsqueeze(0)).squeeze(0)

        action = action[self.action_reorder_mask]
        action = action.detach().numpy()

        target_joint_positions = self.nominal_joint_positions + self.scaling_factor * action

        hard_lower_diff = -(self.joint_positions[self.joint_ids_to_lock] - self.hard_joint_lower_limits[self.joint_ids_to_lock])
        hard_upper_diff = self.joint_positions[self.joint_ids_to_lock] - self.hard_joint_upper_limits[self.joint_ids_to_lock]

        hard_out_of_lower_limits = hard_lower_diff > 0
        hard_out_of_upper_limits = hard_upper_diff > 0
        hard_out_of_limits = hard_out_of_lower_limits | hard_out_of_upper_limits

        p_gains = np.ones(12) * self.nn_p_gain
        d_gains = np.ones(12) * self.nn_d_gain

        p_gains[self.joint_ids_to_lock] = np.where(hard_out_of_limits, self.high_p_gain[self.joint_ids_to_lock], p_gains[self.joint_ids_to_lock])
        d_gains[self.joint_ids_to_lock] = np.where(hard_out_of_limits, self.high_d_gain[self.joint_ids_to_lock], d_gains[self.joint_ids_to_lock])

        # target_joint_positions[self.joint_ids_to_lock] = np.clip(target_joint_positions[self.joint_ids_to_lock], self.hard_joint_lower_limits[self.joint_ids_to_lock], self.hard_joint_upper_limits[self.joint_ids_to_lock])

        low_cmd = deepcopy(self.default_low_cmd)
        for i in range(12):
            low_cmd.motor_cmd[i].q = target_joint_positions[i]
            low_cmd.motor_cmd[i].kp = self.nn_p_gain
            low_cmd.motor_cmd[i].kd = self.nn_d_gain
        
        low_cmd.crc = get_crc(low_cmd)

        import time
        now = time.time()
        if hasattr(self, "last_publish_time_wall"):
            print("dt:", now - self.last_publish_time_wall)
        self.last_publish_time_wall = now

        self.publisher.publish(low_cmd)
        
        self.previous_action = action
    

    def load_policy(self, model_file_name):
        current_path = os.path.dirname(__file__)
        checkpoint_dir = os.path.join(current_path, "policies")
        model_path = os.path.join(checkpoint_dir, model_file_name)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        actor_state_dict = {k.replace("actor.", "", 1): v for k, v in state_dict["model_state_dict"].items() if k.startswith("actor.")}

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