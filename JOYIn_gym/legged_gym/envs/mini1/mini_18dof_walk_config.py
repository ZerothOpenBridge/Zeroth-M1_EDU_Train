import numpy
import torch

from legged_gym.envs.base.base_task_config import BaseConfig
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

USE_PHASE = True


class walk_18dof_MINI1MainBodyCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        use_phase = USE_PHASE
        num_envs = 8192
        episode_length_s = 20  # episode length in seconds

        # actor obs in mini1_code.compute_observation_profile:
        # commands(3) + ang_vel(3) + gravity(3) + dof_pos(14) + dof_vel(14) + actions(14) + phase(2)
        num_obs = 45 + 2*USE_PHASE

        # if not None a priviledge_obs_buf will be returned by step()
        # (critic obs for assymetric training).
        # None is returned otherwise.
        # privileged obs = obs_buf + base_lin_vel(3) + base_height(1) + feet_contact(2)
        # + feet_height(2) + avg_feet_speed_xy(4) + surround_heights(121)
        num_pri_obs = 178 + 18 + 6 + 2*USE_PHASE

        num_actions = 14 + 4  # 12dof + 2dof(shoulder_pitch)
        obs_version = "config1"  # "config1" 或 "config2"
        use_stack = True
        num_stack = 5  # clamp
        obs_profile = "default"

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh

        horizontal_scale = 0.01  # [m]
        vertical_scale = 0.001  # [m]
        border_size = 1  # [m]

        curriculum = False
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 1  # number of terrain cols (types)
        max_init_terrain_level = 2  # maximum initial terrain level

        static_friction = 0.60  # 0.35  # 0.25
        dynamic_friction = 0.60  # 0.35  # 0.25
        restitution = 0.0  # 0.0: no bounce

        # rough terrain only: 1mx1m rectangle (without center line)
        measure_heights = True
        measured_points_x = [-0.3, -0.25, -0.2, -0.15, -0.1, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
        measured_points_y = [-0.3, -0.25, -0.2, -0.15, -0.1, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3]

        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain

        # terrain types:
        # [
        # smooth plane, rough plane,
        # smooth slope, rough slope,
        # smooth wave, rough wave,
        # smooth stairs, rough stairs,
        # stones, discrete,
        # gap, pit,
     # ]
        terrain_proportions = [
            0.0,1.0,
            0.0,0.0,
            0.0,0.0,
            0.0,0.0,
            0.0,0.0,
            0.0,0.0,
        ]

        # trimesh only:
        slope_threshold = 1  # slopes above this threshold will be corrected to vertical surfaces

        terrain_length = 5.0
        terrain_width = 5.0


    class asset(LeggedRobotCfg.asset):
        # 这里必须修改路径，否则无法正确索引到urdf文件
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini1_0/urdf/mini1.urdf"
        name = "mini1"

        # for both joint and link name
        base_name = "base"
        torso_name = "base"  # humanoid pelvis part
        chest_name = "waist"  # humanoid chest part
        forehead_name = "head"  # humanoid head part

        # imu
        imu_name = "imu"

        # waist (for link and joint, with _link and _joint)
        waist_name = "waist"
        waist_yaw_name = "waist_yaw"
        waist_roll_name = "waist_roll"
        waist_pitch_name = "waist_pitch"

        # head (for link and joint, with _link and _joint)
        head_name = "head"
        head_roll_name = "head_roll"
        head_pitch_name = "head_pitch"
        head_yaw_name = "head_yaw"

        # leg (for link, with _link)
        thigh_name = "hip_pitch"
        shank_name = "knee"
        foot_name = "ankle_roll"
        sole_name = "sole"

        # arm (for link, with _link)
        upper_arm_name = "upper_arm"
        lower_arm_name = "lower_arm"
        hand_name = "hand"
        end_effector_name = "end_effector"

        # payload
        payload_name = "payload"

        # leg (for joint, with _joint)
        hip_name = "hip"
        hip_roll_name = "hip_roll"
        hip_pitch_name = "hip_pitch"
        hip_yaw_name = "hip_yaw"
        knee_name = "knee"
        knee_pitch_name = "knee_pitch"
        ankle_name = "ankle"
        ankle_roll_name = "ankle_roll"
        ankle_pitch_name = "ankle_pitch"

        # arm (for joint, with _joint)
        shoulder_name = "shoulder"
        shoulder_roll_name = "shoulder_roll"
        shoulder_pitch_name = "shoulder_pitch"
        shoulder_yaw_name = "shoulder_yaw"
        elbow_name = "elbow"
        wrist_name = "wrist"
        wrist_roll_name = "wrist_roll"
        wrist_pitch_name = "wrist_pitch"
        wrist_yaw_name = "wrist_yaw"

        main_body_joint_names = [waist_name, hip_name, knee_name, ankle_name]

        # for arm reaching
        arm_base_name = "arm_base"
        arm_end_name = "arm_end"

        penalize_contacts_on = ["ankle_roll" ]
        terminate_contacts_on = [
            base_name,
            torso_name,
            thigh_name,
        ]
        collapse_fixed_joints = False  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">

        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation

        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        foot_thickness = 0.03

    class dof(LeggedRobotCfg.dof):
        """
        friction: 关节的摩擦系数，无量纲
        armature: 关节的转子惯性，单位是 kg*m^2
        mechanism: 传动机构的摩擦系数，无量纲
        """
        friction = {
            # left leg
            "left_leg_hip_pitch_joint": 0.0342,
            "left_leg_hip_roll_joint": 0.001,
            "left_leg_hip_yaw_joint": 0.0673,
            "left_leg_knee_pitch_joint": 0.0342,
            "left_leg_ankle_roll_joint": 0.1337,
            "left_leg_ankle_pitch_joint": 0.001,
            # right leg
            "right_leg_hip_pitch_joint": 0.0342,
            "right_leg_hip_roll_joint": 0.001,
            "right_leg_hip_yaw_joint": 0.0673,
            "right_leg_knee_pitch_joint": 0.0342,
            "right_leg_ankle_roll_joint": 0.1337,
            "right_leg_ankle_pitch_joint": 0.001,
            # left arm
            "left_arm_shoulder_pitch_joint": 0.0342,
            "left_arm_shoulder_roll_joint": 0.0342,
            "left_arm_elbow_roll_joint": 0.0342,
            # right arm
            "right_arm_shoulder_pitch_joint": 0.0342,
            "right_arm_shoulder_roll_joint": 0.0342,
            "right_arm_elbow_roll_joint": 0.0342,
        }
        armature = {
            # left leg
            "left_leg_hip_pitch_joint": 0.0342,
            "left_leg_hip_roll_joint": 0.001,
            "left_leg_hip_yaw_joint": 0.105,
            "left_leg_knee_pitch_joint": 0.001,
            "left_leg_ankle_roll_joint": 0.001,
            "left_leg_ankle_pitch_joint": 0.001,
            # right leg
            "right_leg_hip_pitch_joint": 0.0342,
            "right_leg_hip_roll_joint": 0.001,
            "right_leg_hip_yaw_joint": 0.105,
            "right_leg_knee_pitch_joint": 0.001,
            "right_leg_ankle_roll_joint": 0.001,
            "right_leg_ankle_pitch_joint": 0.001,
            # left arm
            "left_arm_shoulder_pitch_joint": 0.001,
            "left_arm_shoulder_roll_joint": 0.001,
            "left_arm_elbow_roll_joint": 0.001,
            # right arm
            "right_arm_shoulder_pitch_joint": 0.001,
            "right_arm_shoulder_roll_joint": 0.001,
            "right_arm_elbow_roll_joint": 0.001,
        }

        mechanism = {
            # left leg
            "left_leg_hip_pitch_joint": 0.0,
            "left_leg_hip_roll_joint": 0.0,
            "left_leg_hip_yaw_joint": 0.0,
            "left_leg_knee_pitch_joint": 0.0,  # 连杆传动机构
            "left_leg_ankle_roll_joint": 0.0,  # 连杆传动机构
            "left_leg_ankle_pitch_joint": 0.0,  # 连杆传动机构
            # right leg
            "right_leg_hip_pitch_joint": 0.0,
            "right_leg_hip_roll_joint": 0.0,
            "right_leg_hip_yaw_joint": 0.0,
            "right_leg_knee_pitch_joint": 0.0,  # 连杆传动机构
            "right_leg_ankle_roll_joint": 0.0,  # 连杆传动机构
            "right_leg_ankle_pitch_joint": 0.0,  # 连杆传动机构
            # left arm
            "left_arm_shoulder_pitch_joint": 0.0,
            "left_arm_shoulder_roll_joint": 0.0,
            "left_arm_elbow_roll_joint": 0.0,
            # right arm
            "right_arm_shoulder_pitch_joint": 0.0,
            "right_arm_shoulder_roll_joint": 0.0,
            "right_arm_elbow_roll_joint": 0.0,
        }

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.27]  # x,y,z [m]
        pos = [0.0, 0.0, 0.275]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # 生效的曲腿角度
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_leg_hip_pitch_joint": -0.261,
            "left_leg_hip_roll_joint": 0.0,
            "left_leg_hip_yaw_joint": 0,
            "left_leg_knee_pitch_joint": 0.523,
            "left_leg_ankle_roll_joint": 0,
            "left_leg_ankle_pitch_joint": -0.261,

            # right leg
            "right_leg_hip_pitch_joint": -0.261,
            "right_leg_hip_roll_joint": 0.0,
            "right_leg_hip_yaw_joint": 0,
            "right_leg_knee_pitch_joint": 0.523,
            "right_leg_ankle_roll_joint": 0,
            "right_leg_ankle_pitch_joint": -0.261,

            # left arm
            "left_arm_shoulder_pitch_joint": 0.0,
            "left_arm_shoulder_roll_joint": 0.0,
            "left_arm_elbow_roll_joint": 0.0,

            # right arm
            "right_arm_shoulder_pitch_joint": 0.0,
            "right_arm_shoulder_roll_joint": 0.0,
            "right_arm_elbow_roll_joint": 0.0,
        }

    class commands(LeggedRobotCfg.commands):

        gait_patterns = [
            "stand",
            "walk",
        ]

        class ranges(LeggedRobotCfg.commands.ranges):

            lin_vel_x = [-0.15, 0.15]  # min max [m/s]
            lin_vel_y = [-0.05, 0.05]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]

        curriculum = False
        curriculum_chg_lin_vel_x = 0.05  # additional linear velocity in x direction
        curriculum_chg_lin_vel_y = 0.0  # additional linear velocity in y direction
        curriculum_chg_ang_vel_yaw = 0.05  # additional angular velocity around z axis

        curriculum_min_lin_vel_x = -0.2  # maximum linear velocity in x direction
        curriculum_min_lin_vel_y = -0.05  # maximum linear velocity in y direction
        curriculum_min_ang_vel_yaw = -0.6  # maximum angular velocity around z axis

        curriculum_max_lin_vel_x = +0.2  # maximum linear velocity in x direction
        curriculum_max_lin_vel_y = +0.05  # maximum linear velocity in y direction
        curriculum_max_ang_vel_yaw = +0.6  # maximum angular velocity around z axis

        max_curriculum = 5.0
        resample_command_interval_s = (
            10.0  # -1: not resample, >0: time before command are changed[s]
        )

        zero_command_prob = 0.0
        threshold_command_lin_vel = 0.01
        threshold_command_ang_vel = 0.05

    class control(LeggedRobotCfg.control):
        action_names = [
            # leg
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_roll",
            "ankle_pitch",
            # arm
            "shoulder_pitch",
            "shoulder_roll",
            "elbow_roll",
        ]

        action_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,
            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "elbow_roll": 1,
        }

        control_type = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,
            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "elbow_roll": 1,
        }

        # kp
        stiffness = {
            # leg
            "hip_pitch": 50,
            "hip_roll": 50,
            "hip_yaw": 50,
            "knee_pitch": 50,
            "ankle_roll": 50,
            "ankle_pitch": 50,
            # arm
            "shoulder_pitch": 50,
            "shoulder_roll": 50,
            "elbow_roll": 50,
        }  # [N*m/rad]
        # kd
        damping = {
            # leg
            "hip_pitch": 2,
            "hip_roll": 2,
            "hip_yaw": 2,
            "knee_pitch": 2,
            "ankle_roll": 2,
            "ankle_pitch": 2,
            # arm
            "shoulder_pitch": 2.0,
            "shoulder_roll": 2.0,
            "elbow_roll": 2.0,
        }  # [N*m*s/rad]

        dof_pos_offset_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,
            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "elbow_roll": 1,
        }

        dof_vel_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,
            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "elbow_roll": 1,
        }

        dof_tor_scale = {
            # leg
            "hip_pitch": 1,
            "hip_roll": 1,
            "hip_yaw": 1,
            "knee_pitch": 1,
            "ankle_roll": 1,
            "ankle_pitch": 1,
            # arm
            "shoulder_pitch": 1,
            "shoulder_roll": 1,
            "elbow_roll": 1,
        }

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class domain_rand(LeggedRobotCfg.domain_rand):

        # randomize friction and restitution
        randomize_friction = True
        friction_range = [0.35, 1.65]  # [0.35, 1.65]  # [0.25, 1.75]

        # randomize base center of mass
        randomize_base_com = True
        add_base_com_range_x = [-0.02, -0.00]  # unit : m 实际机器人质心偏后，故增加向后随机化
        add_base_com_range_y = [-0.03, 0.03]  # unit : m
        add_base_com_range_z = [-0.01, 0.01]  # unit : m

        # randomize torso center of mass
        randomize_torso_com = True
        add_torso_com_range_x = [-0.01, +0.01]
        add_torso_com_range_y = [-0.01, +0.01]
        add_torso_com_range_z = [-0.01, +0.01]

        randomize_control_delay = False
        control_delay_s_range = [0.01, 0.018]
        control_delay_cycle = 1

        # randomize drag robot
        drag_robots = True
        drag_use_xy = False
        drag_force_keep_random = False

        drag_force_axis = "x"
        drag_force_sign = 1.0

         # 持续时间内固定力和方向 / 持续时间内随机力和方向
        drag_interval_s = 10.0  # unit: second
        drag_keep_s = 2.5 # unit: second
        max_drag_force = 30
        max_drag_xy = 0.6  # unit: N
        max_drag_angle = 1.5  # unit: N
        # --------------------------------------------------------------------

    class rewards(LeggedRobotCfg.rewards):
        velocity_threshold = 0.01
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)

        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 0.95
        soft_dof_tor_limit = 0.95

        base_height_target = 0.265  # 0.24

        # ---------------------------------------------------------------
        # Reward related constants
        # 脚接触地面的力的限制比例（相对于重力）
        contact_force_limit_ratio = 1.0

        stand_still_foot_distance = 0.032  # unit: m

        # feet_distance_too_close = max(stand_still_foot_distance - 0.10, 0.10)  # unit: m
        # feet_distance_y_too_close = max(stand_still_foot_distance / 2.0, 0.10)  # unit: m
        feet_distance_too_close = stand_still_foot_distance  # unit: m
        feet_distance_y_too_close = stand_still_foot_distance  # unit: m
        feet_distance_y_too_far = 0.05  # 0.05
        feet_force_z_close_to_ground_contact_force_limit_ratio = 1.0
        feet_distance_x_too_close = 0.05
        feet_air_time_target = 0.3  # unit: s
        feet_stumble_ratio = 5.0  # ratio = fxy / fz
        base_height_offset_range_limit = 0.01  # unit: m
        # -------------------------added---------------------------------
        feet_target_height = 0.015
        feet_target_height_range = 0.005
        # ---------------------------------------------------------------

        # Reward coefficients
        sigma_ankle_roll_posture_roll = -40
        sigma_stand_still_foot_distance = -10.0 * torch.e

        sigma_feet_distance_too_close = -60.0 * torch.e
        sigma_feet_distance_y_too_close = -60.0 * torch.e
        sigma_feet_distance_y_too_far = -60.0 * torch.e

        sigma_feet_speed_z_close_to_ground = -10.0
        sigma_feet_speed_xy_close_to_ground = -10.0  # -10
        sigma_feet_force_z_close_to_ground = -0.05 * torch.e  # -0.01e

        sigma_feet_air_time = -3.0 * torch.e

        sigma_collision = -1.0 * torch.e

        sigma_stand_still = -1.0 * torch.e  # -1e
        sigma_stand_still_dof_pos = -1.0 * torch.e

        sigma_cmd_diff_base_lin_vel_x = -6.0 * torch.e * (1.0 / 0.50)
        sigma_cmd_diff_base_lin_vel_y = -12.0 * torch.e * (1.0 / 1)  # （1、1）
        sigma_cmd_diff_base_ang_vel_yaw = -6.0 * torch.e * (1.0 / 3.00)

        sigma_base_lin_vel_z = -1.0 * torch.e

        sigma_base_height_offset = -10.0 * torch.e
        sigma_base_orient_offset = -1.0 * torch.e

        sigma_base_flat_orient = -5.0 * torch.e
        sigma_torso_flat_orient = -5.0 * torch.e

        sigma_action_diff = -0.1
        sigma_action_diff_diff = -1.0

        sigma_dof_pos_offset = -0.1
        sigma_dof_pos_ankle_offset = -1
        sigma_dof_vel = -0.01
        sigma_dof_acc = -0.005 * torch.e  # reward calculate: 0 ~ 1.0
        sigma_dof_tor = -0.1 * torch.e  # reward calculate: 0 ~ 1.0

        sigma_limits_action = -1.0
        sigma_limits_dof_pos = -1.0
        sigma_limits_dof_vel = -10.0
        sigma_limits_dof_tor = -0.1

        sigma_feet_stumble = -10.0  # -1
        # ---------- added
        sigma_feet_swing_height = -30 * torch.e

        target_joint_pos_scale = 0.7
        period = 1.2
        offset = 0.5


        class scales(LeggedRobotCfg.rewards.scales):

            cmd_diff_base_lin_vel_x = 1.5
            cmd_diff_base_lin_vel_y = 0.5
            cmd_diff_base_ang_vel_yaw = 1.5

            track_vel_hard = 0.2

            base_lin_vel_z = 0.25
            base_acc = 0.25
            base_height_offset_range = 0.1
            base_flat_orient = 0.75

            """
            dof related
            """
            action_diff = -5.00
            action_diff_diff = -1.0
            dof_pos_offset = 0.50
            dof_pos_offset_ankle = 0.25
            hip_yaw_roll_joint_pos = 1
            dof_acc = -0.3
            dof_tor = -0.05

            limits_dof_pos_without_ankle = -10.00
            limits_dof_vel_without_ankle = -5.00
            limits_dof_tor = -1.00

            ankle_roll_posture_roll = 2

            """
            feet related
            """
            feet_speed_xy_close_to_ground = 2.0
            feet_speed_z_close_to_ground = 2.0
            feet_force_z_close_to_ground = -0.50
            feet_stumble = -0.20
            """
            feet_distance related
            """
            feet_distance_too_close = -0.50
            feet_distance_y_too_far = -0.4

            feet_air_time = 2
            feet_swing_height_exp = 2
            contact = 1 if USE_PHASE else 0
            ground_parallel = 0.2
            stand_still = 1
            collision = -1
            shoule_joint_pos = 0.1
            stand_still_1 = 10.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            """
            全部为基本国际单位制

            base_pos: 位置 m
            base_ang: 角度 rad
            action: 动作 rad
            lin_vel: 线速度 m/s
            ang_vel: 角速度 rad/s
            gravity: 重力 m/s^2
            dof_pos: 关节位置 rad
            dof_vel: 关节速度 rad/s
            height_measurements: 高度 m
            """

            ang_vel = 0.1
            lin_vel = 0.02
            gravity = 0.07  # 0.03
            dof_pos = 0.05  # 0.04
            dof_vel = 0.5  # 0.2
            height_measurements = 0.005

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            """
            全部为基本国际单位制

            base_pos: 位置 m
            base_ang: 角度 rad
            action: 动作 rad
            lin_vel: 线速度 m/s
            ang_vel: 角速度 rad/s
            gravity: 重力 m/s^2
            dof_pos: 关节位置 rad
            dof_vel: 关节速度 rad/s
            height_measurements: 高度 m
            """

            lin_vel = 4.0
            ang_vel = 1.5
            omega = 1
            dof_vel = 0.1
            quat = 1.0
            height_measurements = 30

    # viewer camera:
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [3.0, 0.0, 1.6]  # 比 lookat 高 1m
        lookat = [2.0, 0.0, 0.6]  # 对准机器人躯干中心


class walk_18dof_MINI1MainBodyCfgPPO(LeggedRobotCfgPPO):

    runner_class_name = "OnPolicyRunner"

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 64  # per iteration
        max_iterations = 5000  # number of policy updates
        init_at_random_ep_len = True  # if true, the first episode length is random

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = "mini1_18dof_walk"  # 日志保存的名称=蹲姿行走+双臂摆动
        run_name = ""

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint


    class algorithm(LeggedRobotCfgPPO.algorithm):

        num_learning_epochs = 8
        num_mini_batches = 25  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-4  # 5.e-4
        learning_rate_min = 1.0e-5
        learning_rate_max = 1.0e-3
        # entropy_coef=0.03
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.05
        class_name = "PPO"

        # num_mini_batches = 4
        # storage class
        storage_class = "RolloutStorage"

    class policy(LeggedRobotCfgPPO.policy):

        actor_hidden_dims = [ 512, 256, 128]
        critic_hidden_dims = [ 768, 256, 128]

        fixed_std = False

        k = 2
        init_noise_std = [
            0.05 * k,
            0.01 * k,
            0.01 * k,

            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k,

            0.05 * k,
            0.01 * k,
            0.01 * k,

            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k,
            0.1 * k]

        decay_std = False
        decay_ratio = 1 - 2.0e-6
        decay_std_min = 0.1
