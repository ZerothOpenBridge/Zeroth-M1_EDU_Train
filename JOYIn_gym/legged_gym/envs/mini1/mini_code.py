from isaacgym.torch_utils import *

# from legged_gym.envs.mini_base.legged_robot_minibase_bipedal_code import LeggedRobotMINIBASEBipedal
from legged_gym.envs.base.legged_robot_code import LeggedRobot
from pdb import set_trace as st
import time
from isaacgym import gymtorch, gymapi, gymutil

# 导入奖励函数
from legged_gym.envs.base.rewards import Rewards as reward_class


class MINI1_APPEARANCE(LeggedRobot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        # st()
        self.rewards = reward_class(self)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.random_half_phase = torch.randint(0, 2,
            (1, self.num_envs), device=self.device, dtype=torch.long)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        # st()

    def _calculate_and_visualize_com(self, props, env_id):
        """计算并可视化整个机器人的质心"""

        total_mass = 0.0
        com_sum = np.zeros(3)  # [x, y, z]

        # 计算总质心（局部坐标系）
        for i, prop in enumerate(props):
            mass = prop.mass
            com = np.array([prop.com.x, prop.com.y, prop.com.z])

            total_mass += mass
            com_sum += mass * com

        # 整体质心（局部坐标系）
        if total_mass > 0:
            overall_com_local = com_sum / total_mass
        else:
            overall_com_local = np.zeros(3)

        # 获取机器人的世界坐标和姿态
        root_pos = self.root_states[env_id, :3].cpu().numpy()
        root_quat = self.root_states[env_id, 3:7].cpu().numpy()

        # 将局部质心转换到世界坐标系
        overall_com_world = self._transform_local_to_world(overall_com_local, root_pos, root_quat)

        print(
            f"环境 {env_id} - 局部质心: [{overall_com_local[0]:.4f}, {overall_com_local[1]:.4f}, {overall_com_local[2]:.4f}]"
        )
        print(
            f"环境 {env_id} - 世界质心: [{overall_com_world[0]:.4f}, {overall_com_world[1]:.4f}, {overall_com_world[2]:.4f}], 总质量: {total_mass:.4f}kg"
        )

        # 保存世界坐标系下的质心
        self.overall_com = overall_com_world

        return overall_com_world, total_mass

    def _transform_local_to_world(self, local_pos, root_pos, root_quat):
        """将局部坐标转换到世界坐标系"""
        # 使用四元数旋转
        from scipy.spatial.transform import Rotation as R

        # 创建旋转对象
        rotation = R.from_quat(
            [root_quat[1], root_quat[2], root_quat[3], root_quat[0]]
        )  # [x, y, z, w] -> [w, x, y, z]

        # 旋转局部坐标
        rotated_pos = rotation.apply(local_pos)

        # 加上基座位置
        world_pos = root_pos + rotated_pos

        return world_pos

    def _visualize_com(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        com_sphere_geom = gymutil.WireframeSphereGeometry(
            radius=0.04, num_lats=6, num_lons=6, pose=None, color=(1, 0, 0)
        )

        # 计算世界坐标系下的质心
        com_pos_world = self._calculate_and_visualize_com(self.body_props_debug, 0)[0]

        # 绘制质心球体
        sphere_pose = gymapi.Transform(
            gymapi.Vec3(com_pos_world[0], com_pos_world[1], com_pos_world[2]), r=None
        )
        gymutil.draw_lines(
            com_sphere_geom, self.gym, self.viewer, self.env_handles_debug, sphere_pose
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.random_half_phase[0, env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)

    def _drag_robots_use_xy(self):

        self.gym.clear_lines(self.viewer)
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.drag_flag:
            max_xy = self.cfg.domain_rand.max_drag_xy
            max_angle = self.cfg.domain_rand.max_drag_angle
            self.rand_push_force[:, :2] = torch_rand_float(
                -max_xy, max_xy, (self.num_envs, 2), device=self.device
            )  # lin vel x/y

            self.rand_push_torque = torch_rand_float(
                -max_angle, max_angle, (self.num_envs, 3), device=self.device
            )


        if self.cfg.domain_rand.drag_force_keep_random:
            self.drag_flag = 0

        self.root_states[:, 7:9] += self.rand_push_force[:, :2]
        self.root_states[:, 10:13] += self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states)
        )
        #print(self.rand_push_force[:, :2] )
        #print(self.rand_push_torque[:, :3])

    def compute_observation_profile(self):
        selector_walk = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        selector_walk[self.env_ids_of_walk_command] = 1

        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        mask = selector_walk.unsqueeze(1)
        sin_phase *= mask
        cos_phase *= mask

        # st()
        # self.base_projected_gravity[:,0:3]=torch.tensor([0,0,-1]).to(self.device)
        obs_version = getattr(self.cfg.env, "obs_version", "config1")

        if obs_version == "config1":
            # config1: ang_vel 全3维，无 dof_vel
            obs_buf = torch.cat(
                (
                    # command
                    self.commands[:, 0:3] * self.commands_scale,          # 3
                    # base related
                    self.base_ang_vel * self.obs_scales.ang_vel,           # 3
                    self.base_projected_gravity * self.obs_scales.gravity, # 3
                    # dof related
                    self.dof_pos_offset * self.obs_scales.dof_pos,         # 12
                    # action related
                    self.actions * self.obs_scales.action,                 # 12
                ),
                dim=-1,
            )
        elif obs_version == "config2":  # config2
            # config2: 仅 yaw 角速度，含 dof_vel
            obs_buf = torch.cat(
                (
                    # command
                    self.commands[:, 0:3] * self.commands_scale,              # 3
                    # base related
                    self.base_ang_vel[:, 2:3] * self.obs_scales.ang_vel,      # 1
                    self.base_projected_gravity * self.obs_scales.gravity,    # 3
                    # dof related
                    self.dof_pos_offset * self.obs_scales.dof_pos,            # 12
                    self.dof_vel * self.obs_scales.dof_vel,                   # 12
                    # action related
                    self.actions * self.obs_scales.action,                    # 12
                ),
                dim=-1,
            )

        # st()
        if self.cfg.env.use_phase:
            obs_buf = torch.cat((obs_buf, sin_phase, cos_phase), dim=-1)
        # st()
        # print(self.base_lin_vel)
        pri_obs_parts = [
            obs_buf,
            # base related
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_heights_offset * self.obs_scales.height_measurements,
            # foot related
            self.feet_contact,
            self.feet_height * self.obs_scales.height_measurements,
            # self.l_feet_height * self.obs_scales.height_measurements,
            # self.r_feet_height * self.obs_scales.height_measurements,
            self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
            self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
            self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
            self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,
            # selector_walk.unsqueeze(-1),
            # terrain related
            self.surround_heights_offset * self.obs_scales.height_measurements,  # 121维度
        ]
        if obs_version == "config1":
            pri_obs_parts.append(self.dof_vel * self.obs_scales.dof_vel)
            pri_obs_parts.append(self.rand_push_torque )
            pri_obs_parts.append(self.rand_push_force)
        pri_obs_buf = torch.cat(pri_obs_parts, dim=-1)
        # print(self.base_lin_vel[:,0])
        # st()
        # print(self.surround_heights_offset)
        # obs_buf=torch.cat((obs_buf,self.surround_heights_offset*self.obs_scales.height_measurements),dim=-1)
        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def _create_envs(self):
        super()._create_envs()
        left_leg_group_names = ["front_site", "back_site", "left_site", "right_site"]
        right_leg_group_names = ["front_site_r", "back_site_r", "left_site_r", "right_site_r"]
        # st()
        self.left_leg_group_indices = [13, 14, 15, 16]
        self.right_leg_group_indices = [26, 27, 28, 29]

    def _init_buffers(self):
        self.is_stance = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        super()._init_buffers()
        self.last_dof_vel_indeci = self.dof_vel.clone()
        self.initial_p_gains = self.p_gains.clone()

    def post_physics_step_update_state(self):
        # st()
        # self._visualize_com()
        super().post_physics_step_update_state()
        period = self.cfg.rewards.period
        offset = self.cfg.rewards.offset

        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase =  ( self.phase + 0.5 * self.random_half_phase[0] ) %1
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat(
            [self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1
        )
        self.is_stance=self.leg_phase < 0.6
        obs_version = getattr(self.cfg.env, "obs_version", "config1")
        if obs_version == "config1":
            self.compute_ref_state()

        # self._draw_debug_vis()

    def compute_ref_state(self):
        phase = self.phase
        is_swing = self.leg_phase >= 0.6

        sin_pos = torch.sin( torch.pi  * (1 - self.leg_phase) * is_swing / 0.4)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        #self.ref_dof_pos[:, 0] = 1.0*self.dof_pos_offset[:,14] - 1.0*self.dof_pos_offset[:,5]
        self.ref_dof_pos[:, 0] = 0
        self.ref_dof_pos[:, 1] = 0
        self.ref_dof_pos[:, 2] = 0

        self.ref_dof_pos[:, 3] = 0
        self.ref_dof_pos[:, 4] = 0
        self.ref_dof_pos[:, 5] = -sin_pos[:,0] * scale_1
        self.ref_dof_pos[:, 6] = sin_pos[:,0] * scale_2
        self.ref_dof_pos[:, 7] = -sin_pos[:,0] * scale_1
        self.ref_dof_pos[:, 8] = 0

        #self.ref_dof_pos[:, 9] = 1.0*self.dof_pos_offset[:,5] - 1.0*self.dof_pos_offset[:,14]
        self.ref_dof_pos[:, 9] =  0
        self.ref_dof_pos[:, 10] = 0
        self.ref_dof_pos[:, 11] = 0

        self.ref_dof_pos[:, 12] = 0
        self.ref_dof_pos[:, 13] = 0
        self.ref_dof_pos[:, 14] = -sin_pos[:,1] * scale_1
        self.ref_dof_pos[:, 15] = sin_pos[:,1] * scale_2
        self.ref_dof_pos[:, 16] = -sin_pos[:,1] * scale_1
        self.ref_dof_pos[:, 17] = 0

    def _calculate_feet_height(self):
        self.feet_pos = self.rigid_body_states[:, self.feet_indices][
            :, 0 : len(self.feet_indices), 0:3
        ]  # in world frame
        self.feet_quat = self.rigid_body_states[:, self.feet_indices][
            :, 0 : len(self.feet_indices), 3:7
        ]  # in world frame

        self.feet_elevation = torch.zeros(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        for i in range(len(self.feet_indices)):
            feet_forward = quat_apply(self.feet_quat[:, i], self.forward_vec)
            feet_elevation = torch.asin(feet_forward[:, 2:3])  # in world frame
            feet_elevation *= ~(torch.abs(feet_elevation) < 0.001)  # too small set to 0

            self.feet_elevation[:, i] = feet_elevation.squeeze()

        self.feet_height = torch.zeros(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        # 修改腿部高度计算
        self.l_feet_height = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.r_feet_height = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)

        for i in range(len(self.feet_indices)):
            foot_height = torch.mean(
                self.rigid_body_states[:, self.feet_indices][:, i, 2:3]
                - self.cfg.asset.foot_thickness
                - self.measured_heights,
                dim=1,
            )
            # self.l_feet_height=self.rigid_body_states[:, self.left_leg_group_indices][:, :,2:3].squeeze(-1)
            # self.r_feet_height=self.rigid_body_states[:, self.right_leg_group_indices][:, :,2:3].squeeze(-1)
            self.feet_height[:, i] = foot_height

    def _compute_torques(self, actions):
        """
        Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor, [num_envs, num_dof]): Actions to be converted to torques

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.action_scales

        # expand actions from (num_envs, num_actions) to (num_envs, num_dofs)
        actions_expanded = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32
        )

        for i in range(self.num_actions):
            action_index = self.action_indices[i]
            # st()
            actions_expanded[:, action_index : action_index + 1] = actions_scaled[:, i : i + 1]

        # --------------------------------------------------
        # st()
        control_target = actions_expanded
        # st()
        self.dof_pos_target = control_target.clone()  # save for later use

        # --------------------------------------------------
        dof_acc = (self.dof_vel - self.last_dof_vel_indeci) / self.sim_params.dt
        self.last_dof_vel_indeci[:] = self.dof_vel[:]
        error = control_target + self.default_dof_pos - self.dof_pos

        self.initial_p_gains = self.initial_p_gains.clone()

        self.p_gains = torch.where(
            (dof_acc * error) < 0, 1 * self.initial_p_gains, self.initial_p_gains
        )
        self.p_gains = torch.where(
            torch.ones(
                self.cfg.env.num_envs, self.cfg.env.num_actions, dtype=bool, device=self.device
            ),
            1.0 * self.initial_p_gains,
            self.initial_p_gains,
        )
        # st()
        # print(self.p_gains)
        # st()
        torques = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32
        )

        for i in range(self.num_dofs):
            control_type = self.control_types[i]

            if control_type == 1:  # Position Control (PD controller)
                # st()

                torques_P_p = (
                    self.p_gain_scales[:, i : i + 1]
                    * self.p_gains[:, i : i + 1]
                    * (
                        control_target[:, i : i + 1]
                        + self.default_dof_pos[:, i : i + 1]
                        - self.dof_pos[:, i : i + 1]
                    )
                )
                torques_P_d = (
                    self.d_gain_scales[:, i : i + 1]
                    * self.d_gains[i]
                    * (0 - self.dof_vel[:, i : i + 1])
                )

                torques[:, i : i + 1] = torques_P_p + torques_P_d

            elif control_type == 2:  # Velocity Control (PD controller)
                torques_V_p = 0
                torques_V_d = (
                    self.d_gain_scales[:, i : i + 1]
                    * self.d_gains[i]
                    * (control_target[:, i : i + 1] - self.dof_vel[:, i : i + 1])
                )
                torques[:, i : i + 1] = torques_V_p + torques_V_d

            elif control_type == 3:  # Torque Control
                torques[:, i : i + 1] = control_target[:, i : i + 1]

            else:
                raise NameError(f"Unknown controller type: {control_type}")

        # --------------------------------------------------

        # ratio
        torques *= self.motor_strength_scales

        # clip
        torques = torch.clip(torques, min=-self.dof_tor_limits, max=+self.dof_tor_limits)

        # store torques
        self.torques = torques.clone()

        return torques

    def _update_terrain_curriculum_detection(self, env_ids):
        """
        Detects if the robot should move up or down in the terrain curriculum
        """
        # 1. detect based on distance from origin
        distance = torch.norm(
            self.root_states[env_ids, 0:2] - self.env_origins[env_ids, 0:2], dim=1
        )  # dims 2->1

        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 8
        # st()
        move_up_level = 1

        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, 0:2], dim=1) * self.max_episode_length_s * 0.5
        ) * ~move_up
        move_down_level = 1

        return move_up, move_up_level, move_down, move_down_level

    def get_mirror_observations(self, observations):
        """
        Returns the mirror observations
        """
        if self.cfg.mirror.enable_mirror:
            # st()
            mirror_observations = torch.zeros_like(observations)

            # 如果使用 stack，则需要对每个 stack 进行 mirror
            if self.cfg.env.use_stack:

                # 1. change direction
                for i in range(self.cfg.env.num_stack):
                    mirror_observations[
                        :, self.cfg.env.num_obs * (i + 0) : self.cfg.env.num_obs * (i + 1)
                    ] = observations[
                        :, self.cfg.env.num_obs * (i + 0) : self.cfg.env.num_obs * (i + 1)
                    ] * torch.from_numpy(
                        self.cfg.mirror.observations_coefficient
                    ).float().to(
                        self.device
                    )

                mirror_observations_change_direction = mirror_observations.clone()  # backup

                # 2. exchange sequence
                for i in range(self.cfg.env.num_stack):
                    for idx1, idx2 in self.cfg.mirror.observations_exchange:
                        mirror_observations[:, idx1 + self.cfg.env.num_obs * (i + 0)] = (
                            mirror_observations_change_direction[
                                :, idx2 + self.cfg.env.num_obs * (i + 0)
                            ]
                        )
                        mirror_observations[:, idx2 + self.cfg.env.num_obs * (i + 0)] = (
                            mirror_observations_change_direction[
                                :, idx1 + self.cfg.env.num_obs * (i + 0)
                            ]
                        )

            else:
                # 1. change direction
                mirror_observations = observations * torch.from_numpy(
                    self.cfg.mirror.observations_coefficient
                ).float().to(self.device)

                mirror_observations_change_direction = mirror_observations.clone()  # backup

                # 2. exchange sequence
                for idx1, idx2 in self.cfg.mirror.observations_exchange:
                    mirror_observations[:, idx1] = mirror_observations_change_direction[:, idx2]
                    mirror_observations[:, idx2] = mirror_observations_change_direction[:, idx1]

        else:
            mirror_observations = observations

        return mirror_observations

    def get_mirror_actions(self, actions):
        """
        Returns the mirror actions
        """
        if self.cfg.mirror.enable_mirror:
            mirror_actions = torch.zeros_like(actions)
            # st()
            # 1. change direction
            mirror_actions = actions * torch.from_numpy(
                self.cfg.mirror.actions_coefficient
            ).float().to(self.device)

            mirror_actions_change_direction = mirror_actions.clone()  # backup

            # 2. exchange sequence
            for idx1, idx2 in self.cfg.mirror.actions_exchange:
                mirror_actions[:, idx1] = mirror_actions_change_direction[:, idx2]
                mirror_actions[:, idx2] = mirror_actions_change_direction[:, idx1]
        else:
            mirror_actions = actions

        return mirror_actions

    def _reward_termination(self):
        # Terminal reward / penalty
        reward_termination = self.reset_buf * ~self.time_out_buf
        return reward_termination

    def _resample_commands(self, env_ids=None, command_profile=None):
        """
        Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
            command_profile (str): Command profile to be used
        """
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, step=1, device=self.device)

        if command_profile is None:
            command_profile = self.cfg.commands.command_profile

        if command_profile == "not_use":
            pass

        if command_profile == "base_velocity":
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                         self.command_ranges["lin_vel_x"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                         self.command_ranges["lin_vel_y"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

            zero_prob = getattr(self.cfg.commands, "zero_command_prob", 0.0)
            if zero_prob > 0 and len(env_ids) > 0:
                stand_mask = torch.rand(len(env_ids), device=self.device) < zero_prob
                stand_ids = env_ids[stand_mask]
                if len(stand_ids) > 0:
                    self.commands[stand_ids, 0:3] = 0.0

            if self.cfg.commands.heading_command:
                self.commands_base_heading[env_ids] = torch_rand_float(self.command_ranges["heading"][0],
                                                                       self.command_ranges["heading"][1],
                                                                       (len(env_ids), 1),
                                                                       device=self.device).squeeze(1)

            self._command_refinement(env_ids)
            # map commands list to separate commands
            self.commands_base_lin_vel_x = self.commands[:, 0:1]
            self.commands_base_lin_vel_y = self.commands[:, 1:2]
            self.commands_base_ang_vel_yaw = self.commands[:, 2:3]


        self.update_gait_generator_pattern()
