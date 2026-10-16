from isaacgym.torch_utils import *
import torch


class Rewards:
    def __init__(self, env):
        self.env = env
        self.get_all_rewards()

    def get_all_rewards(self):
        pass

    def _reward_cmd_diff_base_lin_vel_x(self):
        """
        奖励基座前向速度与指令接近
        """
        error_x_vel = torch.abs(self.env.commands_base_lin_vel_x - self.env.base_lin_vel[:, 0:1])
        # st()
        error_x_vel = torch.sum(error_x_vel, dim=1)  # dims 2->1

        reward_x_vel = torch.exp(self.env.cfg.rewards.sigma_cmd_diff_base_lin_vel_x * error_x_vel)
        return reward_x_vel

    def _reward_cmd_diff_base_lin_vel_y(self):
        """
        奖励基座侧向速度与指令接近
        """
        error_y_vel = torch.abs(self.env.commands_base_lin_vel_y - self.env.base_lin_vel[:, 1:2])
        error_y_vel = torch.sum(error_y_vel, dim=1)  # dims 2->1

        reward_y_vel = torch.exp(self.env.cfg.rewards.sigma_cmd_diff_base_lin_vel_y * error_y_vel)
        # print(f'cmd_y:{reward_y_vel}')
        return reward_y_vel

    def _reward_cmd_diff_base_ang_vel_yaw(self):
        """
        奖励基座偏航角速度与指令接近
        """
        error_yaw_vel = torch.abs(
            self.env.commands_base_ang_vel_yaw - self.env.base_ang_vel[:, 2:3]
        )
        error_yaw_vel = torch.sum(error_yaw_vel, dim=1)  # dims 2->1

        reward_yaw_vel = torch.exp(
            self.env.cfg.rewards.sigma_cmd_diff_base_ang_vel_yaw * error_yaw_vel
        )
        return reward_yaw_vel

    def _reward_ground_parallel(self):
        """
        通过脚踝周围四个点的协方差，惩罚脚踝不平行于地面
        """
        left_ankle_pos = self.env.rigid_body_states[:, self.env.left_leg_group_indices, 2] * 10
        right_ankle_pos = self.env.rigid_body_states[:, self.env.right_leg_group_indices, 2] * 10
        # st()
        var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
        var = torch.mean(
            torch.concat(
                [left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1
            ),
            dim=-1,
        )
        reward = torch.where(var < 0.004, 1, -10)
        # print("ground_parallel_reward:", var[0])
        # st()
        # print(reward)

        return reward

    def _reward_contact(self):
        """
        鼓励步态符合规定的相位，仅仅奖励运动指令
        """
        res = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        for i in range(2):
            is_stance = self.env.leg_phase[:, i] < 0.6
            contact = self.env.feet_contact[:, i] > 0.1
            # contact=self.env.feet_contact_for_schedule[:,i]
            # print(contact)
            res += ~(contact ^ is_stance)
            # print(contact)

        # print("contact_phase",self.env.leg_phase[:, 0] < 0.55)
        # st()
        selector_walk = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_walk[self.env.env_ids_of_walk_command] = 1
        res *= selector_walk
        # st()
        # print(res)
        return res

    def _reward_stand_still(self):
        condition_both_feet_contact = self.env.feet_contact[:, 0] * self.env.feet_contact[:, 1]
        selector_walk = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_walk[self.env.env_ids_of_walk_command] = 1
        res = ~selector_walk * condition_both_feet_contact
        # st()
        return res

    def _reward_feet_swing_height_exp(self):
        '''
        鼓励抬腿高度在预期范围内
        '''
        is_left_swing = self.env.leg_phase[:, 0] >= 0.6
        is_right_swing = self.env.leg_phase[:, 1] >= 0.6

        left_foot_height = self.env.get_left_foot_height()
        right_foot_height = self.env.get_right_foot_height()
        left_foot_height_offset = torch.abs(left_foot_height - self.env.cfg.rewards.feet_target_height)
        right_foot_height_offset = torch.abs(right_foot_height - self.env.cfg.rewards.feet_target_height)
        left_foot_height_offset = \
            (left_foot_height_offset - self.env.cfg.rewards.feet_target_height_range) \
            * (left_foot_height_offset > self.env.cfg.rewards.feet_target_height_range)
        right_foot_height_offset = \
            (right_foot_height_offset - self.env.cfg.rewards.feet_target_height_range) \
            * (right_foot_height_offset > self.env.cfg.rewards.feet_target_height_range)
        reward_left_foot_height = torch.sum(left_foot_height_offset, dim=1)
        reward_right_foot_height = torch.sum(right_foot_height_offset, dim=1)

        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][:, 0, 1]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 1
        ]
        foot_distance_y = left_foot_pos_in_world_frame - right_foot_pos_in_world_frame
        # st()

        # print(left_foot_height)
        # st()
        reward = ~self.env.feet_contact[:, 0] * is_left_swing * torch.exp(
            self.env.cfg.rewards.sigma_feet_swing_height * reward_left_foot_height) + ~self.env.feet_contact[
            :, 1] * is_right_swing * torch.exp(self.env.cfg.rewards.sigma_feet_swing_height * reward_right_foot_height)
        # print(reward)
        idx = torch.where(foot_distance_y > 0.06)
        # print(idx)
        # st()
        reward[idx] = 0

        selector_walk = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_walk[self.env.env_ids_of_walk_command] = 1
        reward *= selector_walk

        # print(reward)
        return reward

    def _reward_base_lin_vel_z(self):
        """
        鼓励基座垂直速度为0
        """
        error_z_vel = torch.abs(0 - self.env.base_lin_vel[:, 2:3])
        error_z_vel = torch.sum(error_z_vel, dim=1)  # dims 2->1

        reward_z_vel = torch.exp(self.env.cfg.rewards.sigma_base_lin_vel_z * error_z_vel)
        return reward_z_vel

    def _reward_base_height_offset_range(self):
        """
        鼓励基座高度保持在一定范围内
        """
        error_base_height = torch.abs(self.env.base_heights_offset)

        # 只计算高度超出范围的情况
        error_base_height = (
                                    error_base_height - self.env.cfg.rewards.base_height_offset_range_limit
                            ) * (error_base_height > self.env.cfg.rewards.base_height_offset_range_limit)
        error_base_height = torch.sum(error_base_height, dim=1)  # dims 2->1
        # print(error_base_height)
        reward_base_height = torch.exp(
            self.env.cfg.rewards.sigma_base_height_offset * error_base_height
        )
        return reward_base_height

    def _reward_base_flat_orient(self):
        """
        鼓励基座水平
        """
        base_projected_gravity = self.env.base_projected_gravity

        error_base_flat_orient = torch.abs(base_projected_gravity[:, 0:2])
        error_base_flat_orient = torch.sum(error_base_flat_orient, dim=1)  # dims 2->1

        reward_base_flat_orient = torch.exp(
            self.env.cfg.rewards.sigma_base_flat_orient * error_base_flat_orient
        )
        return reward_base_flat_orient

    def _reward_action_diff(self):
        """
        惩罚动作剧烈变化，鼓励动作平滑
        """
        error_action_diff = (self.env.actions - self.env.last_actions) * self.env.action_scales

        error_action_diff = torch.abs(error_action_diff)
        error_action_diff = torch.sum(error_action_diff, dim=1)  # dims 2->1

        reward_action_diff = 1 - torch.exp(
            self.env.cfg.rewards.sigma_action_diff * error_action_diff
        )
        return reward_action_diff

    def _reward_action_diff_diff(self):
        """
        惩罚动作二阶变化率，进一步鼓励动作平滑
        """
        error_action_diff = (self.env.actions - self.env.last_actions) * self.env.action_scales
        error_action_diff_last = (
                                         self.env.last_actions - self.env.last_last_actions
                                 ) * self.env.action_scales

        error_action_diff_diff = torch.abs(error_action_diff - error_action_diff_last)
        error_action_diff_diff = torch.sum(error_action_diff_diff, dim=1)  # dims 2->1

        reward_action_diff_diff = 1 - torch.exp(
            self.env.cfg.rewards.sigma_action_diff_diff * error_action_diff_diff
        )
        return reward_action_diff_diff

    def _reward_hip_yaw_roll_joint_pos(self):

        selected_columns = [3, 12]
        diff = (self.env.dof_pos - self.env.ref_dof_pos)[:, selected_columns]
        r = (1 - torch.exp(2 * torch.norm(diff, dim=1))).clamp(-1, 0)

        selector_walk = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_walk[(self.env.commands_base_ang_vel_yaw > 0.05).squeeze(-1)] = 0

        r *= selector_walk

        return r

    def _reward_shoule_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """

        selected_columns = [0, 1, 2, 9, 10, 11]
        # selected_columns = [ 1, 2, 10, 11]
        diff = (self.env.dof_pos - (self.env.ref_dof_pos + self.env.default_dof_pos))[:, selected_columns]
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(
            diff, dim=1
        ).clamp(0, 1)

        selector_walk = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_walk[self.env.env_ids_of_walk_command] = 1
        r *= selector_walk

        return r

    def _reward_dof_pos_offset(self):
        """
        鼓励所有关节保持在默认位置
        """
        error_dof_pos_offset = (
                torch.abs(self.env.dof_pos - self.env.default_dof_pos) * self.env.dof_pos_offset_scales
        )
        error_dof_pos_offset = torch.sum(error_dof_pos_offset, dim=1)  # dims 2->1

        reward_dof_pos_offset = torch.exp(
            self.env.cfg.rewards.sigma_dof_pos_offset * error_dof_pos_offset
        )
        # print(reward_dof_pos_offset)
        return reward_dof_pos_offset

    def _reward_dof_pos_offset_ankle(self):
        """
        鼓励ankle_roll保持在默认位置
        """
        error_dof_pos_offset = (
                torch.abs(
                    self.env.dof_pos[:, self.env.ankle_roll_indices]
                    - self.env.default_dof_pos[:, self.env.ankle_roll_indices]
                )
                * self.env.dof_pos_offset_scales[self.env.ankle_roll_indices]
        )
        error_dof_pos_offset = torch.sum(error_dof_pos_offset, dim=1)  # dims 2->1

        reward_dof_pos_offset = torch.exp(
            self.env.cfg.rewards.sigma_dof_pos_ankle_offset * error_dof_pos_offset
        )
        # st()
        return reward_dof_pos_offset

    def _reward_dof_vel(self):
        """
        惩罚关节速度
        """
        error_dof_vel = torch.abs(self.env.dof_vel) * self.env.dof_vel_scales
        error_dof_vel = torch.sum(error_dof_vel, dim=1)  # dims 2->1

        reward_dof_vel = 1 - torch.exp(self.env.cfg.rewards.sigma_dof_vel * error_dof_vel)
        # print(reward_dof_vel,self.env.cfg.rewards.sigma_dof_vel
        #                                * error_dof_vel)
        return reward_dof_vel

    def _reward_dof_acc(self):
        """
        惩罚关节加速度
        """
        error_dof_acc = (
                torch.abs((self.env.dof_vel - self.env.last_dof_vel) / self.env.dt)
                * self.env.dof_vel_scales
        )
        error_dof_acc = torch.sum(error_dof_acc, dim=1)  # dims 2->1

        reward_dof_acc = 1 - torch.exp(self.env.cfg.rewards.sigma_dof_acc * error_dof_acc)
        # print(self.env.cfg.rewards.sigma_dof_acc
        #                                * error_dof_acc)
        return reward_dof_acc

    def _reward_dof_tor(self):
        """
        惩罚关节力矩
        """
        error_dof_tor = torch.abs(self.env.dof_tor) * self.env.dof_tor_scales
        error_dof_tor = torch.sum(error_dof_tor, dim=1)  # dims 2->1

        reward_dof_tor = 1 - torch.exp(self.env.cfg.rewards.sigma_dof_tor * error_dof_tor)
        # print(self.env.cfg.rewards.sigma_dof_tor
        #                                * error_dof_tor)
        return reward_dof_tor

    def _reward_feet_stumble(self):
        """
        惩罚与垂直平面的碰撞
        """
        left_foot_fxy = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices][:, 0, 0:2], dim=1
        ).unsqueeze(1)
        # print(left_foot_fxy)
        # st()
        right_foot_fxy = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices][:, 1, 0:2], dim=1
        ).unsqueeze(1)
        # print(left_foot_fxy)
        left_foot_fz = self.env.contact_forces[:, self.env.feet_indices][:, 0, 2:3]
        right_foot_fz = self.env.contact_forces[:, self.env.feet_indices][:, 1, 2:3]
        # print(left_foot_fz)
        error_left_foot_f = left_foot_fxy - self.env.cfg.rewards.feet_stumble_ratio * torch.abs(
            left_foot_fz
        )
        error_right_foot_f = right_foot_fxy - self.env.cfg.rewards.feet_stumble_ratio * torch.abs(
            right_foot_fz
        )

        error_left_foot_f = error_left_foot_f * (error_left_foot_f > 0)
        error_right_foot_f = error_right_foot_f * (error_right_foot_f > 0)

        error_left_foot_f = torch.sum(error_left_foot_f, dim=1)  # dims 2->1
        error_right_foot_f = torch.sum(error_right_foot_f, dim=1)  # dims 2->1

        reward_left_foot_f = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_stumble * error_left_foot_f
        )
        reward_right_foot_f = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_stumble * error_right_foot_f
        )

        reward_feet_stumble = reward_left_foot_f + reward_right_foot_f

        return reward_feet_stumble

    def _reward_limits_dof_tor(self):
        """
        惩罚关节力矩超出限制
        """
        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.env.num_actions))
        # ----------------------------------------------

        error_limits_dof_tor = (
                torch.abs(self.env.dof_tor[:, related_indexes])
                - self.env.soft_dof_tor_limits[related_indexes]
        ).clip(min=0.0)
        error_limits_dof_tor = torch.sum(error_limits_dof_tor, dim=1)  # dims 2->1

        reward_limits_dof_tor = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_dof_tor * error_limits_dof_tor
        )

        return reward_limits_dof_tor

    def _reward_limits_dof_pos_without_ankle(self):
        """
        惩罚关节位置超出限制
        """
        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.env.num_actions))

        # remove ankle joints, because when step on the ground, the position is not controllable
        related_indexes = [i for i in related_indexes if i not in self.env.ankle_indices]
        # ----------------------------------------------
        # st()
        out_of_limits = -(
                self.env.dof_pos[:, related_indexes]
                - self.env.soft_dof_pos_limits[:, 0][related_indexes]
        ).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (
                self.env.dof_pos[:, related_indexes]
                - self.env.soft_dof_pos_limits[:, 1][related_indexes]
        ).clip(
            min=0.0
        )  # upper limit

        error_limits_dof_pos = torch.abs(out_of_limits)
        error_limits_dof_pos = torch.sum(error_limits_dof_pos, dim=1)  # dims 2->1

        reward_limits_dof_pos = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_dof_pos * error_limits_dof_pos
        )

        return reward_limits_dof_pos

    def _reward_limits_dof_vel_without_ankle(self):
        """
        惩罚关节速度超出限制
        """
        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.env.num_actions))

        # remove ankle joints, because when step on the ground, the velocity is not controllable
        related_indexes = [i for i in related_indexes if i not in self.env.ankle_indices]
        # ----------------------------------------------

        error_limits_dof_vel = (
                torch.abs(self.env.dof_vel[:, related_indexes])
                - self.env.soft_dof_vel_limits[related_indexes]
        ).clip(min=0.0)
        error_limits_dof_vel = torch.sum(error_limits_dof_vel, dim=1)  # dims 2->1

        reward_limits_dof_vel = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_dof_vel * error_limits_dof_vel
        )

        return reward_limits_dof_vel

    def _reward_ankle_roll_posture_roll(self):
        """
        鼓励ankle_roll刚体的pitch角和roll角为0
        """
        feet_eular_0 = self.env.get_euler_xyz_tensor(
            self.env.rigid_body_states[:, self.env.feet_indices][:, 0, 3:7]  # in world frame
        )[:, 0:1]
        feet_eular_1 = self.env.get_euler_xyz_tensor(
            self.env.rigid_body_states[:, self.env.feet_indices][:, 1, 3:7]
        )[:, 0:1]
        # print(feet_eular_1.size())
        rew = torch.exp(
            self.env.cfg.rewards.sigma_ankle_roll_posture_roll
            * (torch.norm(feet_eular_0, dim=1) + torch.norm(feet_eular_1, dim=1))
        )
        # print(nn.size())
        return rew

    # def _reward_swing_feet_speed(self):
    def _reward_feet_speed_z_close_to_ground(self):
        """
        在接近地面时，限制脚的 Z 轴速度（垂直速度）越小奖励越高。
        """
        # 获取左右脚当前离地高度与最大高度
        left_foot_height = self.env.get_left_foot_height()
        right_foot_height = self.env.get_right_foot_height()
        left_foot_height_max = self.env.get_left_foot_height_max()
        right_foot_height_max = self.env.get_right_foot_height_max()

        # 接近地面的权重：脚高 < max/4 时，权重从 1（高度=0）到 0（高度=max/4）
        error_left_foot_close_to_ground = (
                torch.abs(left_foot_height - left_foot_height_max / 4)
                * (left_foot_height < left_foot_height_max / 4)
                / (left_foot_height_max / 4)
        )
        error_right_foot_close_to_ground = (
                torch.abs(right_foot_height - right_foot_height_max / 4)
                * (right_foot_height < right_foot_height_max / 4)
                / (right_foot_height_max / 4)
        )

        # 提取 Z 轴速度的绝对值（垂直速度的大小）
        left_foot_speed_z = torch.abs(self.env.avg_feet_speed_xyz[:, 0, 2]).unsqueeze(1)
        right_foot_speed_z = torch.abs(self.env.avg_feet_speed_xyz[:, 1, 2]).unsqueeze(1)

        # 接近地面时的 Z 轴速度误差：垂直速度 × 接近地面权重
        error_left_foot_speed_z = left_foot_speed_z * error_left_foot_close_to_ground
        error_right_foot_speed_z = right_foot_speed_z * error_right_foot_close_to_ground

        # 汇总左右脚误差
        error_feet_speed_z_close_to_ground = (
                error_left_foot_speed_z + error_right_foot_speed_z
        )
        error_feet_speed_z_close_to_ground = torch.sum(
            error_feet_speed_z_close_to_ground, dim=1
        )  # [batch_size]

        # 奖励：误差越大奖励越小（假设 sigma 为负值）
        reward_feet_speed_z_close_to_ground = torch.exp(
            self.env.cfg.rewards.sigma_feet_speed_z_close_to_ground
            * error_feet_speed_z_close_to_ground
        )

        selector_stand = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_stand[self.env.env_ids_of_stand_command] = 1
        reward_feet_speed_z_close_to_ground*=selector_stand

        return reward_feet_speed_z_close_to_ground

    def _reward_feet_speed_xy_close_to_ground(self):
        """
        在接近地面时xy方向速度越小奖励越高
        """
        left_foot_height = self.env.get_left_foot_height()
        right_foot_height = self.env.get_right_foot_height()

        left_foot_height_max = self.env.get_left_foot_height_max()
        right_foot_height_max = self.env.get_right_foot_height_max()

        error_left_foot_close_to_ground = (
                torch.abs(left_foot_height - left_foot_height_max / 4)
                * (left_foot_height < left_foot_height_max / 4)
                / (left_foot_height_max / 4)
        )
        error_right_foot_close_to_ground = (
                torch.abs(right_foot_height - right_foot_height_max / 4)
                * (right_foot_height < right_foot_height_max / 4)
                / (right_foot_height_max / 4)
        )

        error_left_foot_speed_xy_close_to_ground = (
                torch.norm(self.env.avg_feet_speed_xyz[:, 0, 0:2], dim=1).unsqueeze(1)
                * error_left_foot_close_to_ground
        )
        error_right_foot_speed_xy_close_to_ground = (
                torch.norm(self.env.avg_feet_speed_xyz[:, 1, 0:2], dim=1).unsqueeze(1)
                * error_right_foot_close_to_ground
        )

        error_feet_speed_xy_close_to_ground = (
                error_left_foot_speed_xy_close_to_ground + error_right_foot_speed_xy_close_to_ground
        )
        error_feet_speed_xy_close_to_ground = torch.sum(
            error_feet_speed_xy_close_to_ground, dim=1
        )  # dims 2->1

        reward_feet_speed_xy_close_to_ground = torch.exp(
            self.env.cfg.rewards.sigma_feet_speed_xy_close_to_ground
            * error_feet_speed_xy_close_to_ground
        )

        # print(reward_feet_speed_xy_close_to_ground)
        return reward_feet_speed_xy_close_to_ground

    def _reward_feet_air_time(self):
        """
        奖励抬腿高度达到预期时间，每次落地时计算奖励
        """
        left_foot_air_time = self.env.feet_air_time_last[:, 0:1]  # dims 2
        right_foot_air_time = self.env.feet_air_time_last[:, 1:2]  # dims 2

        error_left_foot_air_time = (
                torch.abs(left_foot_air_time - self.env.cfg.rewards.feet_air_time_target)
                / self.env.cfg.rewards.feet_air_time_target
        )  # dims 2
        error_right_foot_air_time = (
                torch.abs(right_foot_air_time - self.env.cfg.rewards.feet_air_time_target)
                / self.env.cfg.rewards.feet_air_time_target
        )  # dims 2

        error_left_foot_air_time = torch.sum(error_left_foot_air_time, dim=1)  # dims 2->1
        error_right_foot_air_time = torch.sum(error_right_foot_air_time, dim=1)  # dims 2->1

        # if error_left_foot_air_time!=1:
        #     print("left_foot_air_time:",error_left_foot_air_time)
        reward_left_foot_air_time = torch.exp(
            self.env.cfg.rewards.sigma_feet_air_time * error_left_foot_air_time
        )
        reward_right_foot_air_time = torch.exp(
            self.env.cfg.rewards.sigma_feet_air_time * error_right_foot_air_time
        )
        # reward_left_foot_air_time*=error_left_foot_air_time<0.7
        # reward_right_foot_air_time*=error_right_foot_air_time<0.7

        # ----------------------------

        condition_left_foot_first_contact_ground = self.env.feet_contact_trig[
            :, 0
        ]  # dims 1, 第一次触地flag
        condition_right_foot_first_contact_ground = self.env.feet_contact_trig[
            :, 1
        ]  # dims 1, 第一次触地flag

        reward_left_foot_air_time *= condition_left_foot_first_contact_ground
        reward_right_foot_air_time *= condition_right_foot_first_contact_ground

        selector_move = torch.ones(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_move[self.env.env_ids_of_off_command] = 0
        selector_move[self.env.env_ids_of_stand_command] = 0

        reward_left_foot_air_time *= selector_move
        reward_right_foot_air_time *= selector_move
        # ----------------------------

        reward_feet_air_time = reward_left_foot_air_time + reward_right_foot_air_time
        # print(reward_left_foot_air_time)
        # st()
        return reward_feet_air_time

    def _reward_feet_force_z_close_to_ground(self):
        """
        惩罚每只脚的受力大于自身的重量, 并乘上了一个权重, 离地面越近权重越大并为1
        """
        left_foot_height = self.env.get_left_foot_height()
        right_foot_height = self.env.get_right_foot_height()

        left_foot_height_max = self.env.get_left_foot_height_max()
        right_foot_height_max = self.env.get_right_foot_height_max()

        error_left_foot_force_z_close_to_ground = torch.clip(
            self.env.contact_forces[:, self.env.feet_indices][:, 0, 2:3]
            - self.env.contact_forces_limit
            * self.env.cfg.rewards.feet_force_z_close_to_ground_contact_force_limit_ratio,
            min=0,
        )
        error_right_foot_force_z_close_to_ground = torch.clip(
            self.env.contact_forces[:, self.env.feet_indices][:, 1, 2:3]
            - self.env.contact_forces_limit
            * self.env.cfg.rewards.feet_force_z_close_to_ground_contact_force_limit_ratio,
            min=0,
        )
        # print(error_left_foot_force_z_close_to_ground)
        error_feet_force_z_close_to_ground = (
                error_left_foot_force_z_close_to_ground + error_right_foot_force_z_close_to_ground
        )
        error_feet_force_z_close_to_ground = torch.sum(
            error_feet_force_z_close_to_ground, dim=1
        )  # dims 2->1

        reward_feet_force_z_close_to_ground = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_force_z_close_to_ground
            * error_feet_force_z_close_to_ground
        )
        # print(self.env.contact_forces[:, self.env.feet_indices][:, 0, 2:3])
        # print(reward_feet_force_z_close_to_ground)
        return reward_feet_force_z_close_to_ground

    def _reward_collision(self):
        """
        惩罚碰撞
        """
        # Penalize collisions on selected bodies
        flag_penalty = (
                torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1)
                > 0.1
        )
        reward_collision = torch.sum(flag_penalty, dim=1)

        return reward_collision

    def _reward_termination(self):
        # Terminal reward / penalty
        reward_termination = self.env.reset_buf * ~self.env.time_out_buf
        return reward_termination

    def _reward_stand_still_1(self):
        # Penalize not standing still
        error_stand_still = (
                torch.abs(self.env.dof_pos - self.env.default_dof_pos_tenors)
                * self.env.dof_pos_offset_scales
        )
        error_stand_still = torch.sum(error_stand_still, dim=1)  # dims 2->1

        reward_stand_still = torch.exp(self.env.cfg.rewards.sigma_stand_still * error_stand_still)

        selector_stand_still = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.bool
        )
        selector_stand_still[self.env.env_ids_of_stand_command] = 1

        reward_stand_still *= selector_stand_still

        return reward_stand_still

    def _reward_torso_flat_orient(self):
        if len(self.env.torso_indices) > 0:
            torso_projected_gravity = quat_rotate_inverse(
                self.env.rigid_body_states[:, self.env.torso_indices][:, 0, 3:7],
                self.env.gravity_vec,
            )

            error_torso_flat_orient = torch.abs(torso_projected_gravity[:, 0:2])
            error_torso_flat_orient = torch.sum(error_torso_flat_orient, dim=1)  # dims 2->1

            reward_torso_flat_orient = torch.exp(
                self.env.cfg.rewards.sigma_torso_flat_orient * error_torso_flat_orient
            )

        else:
            reward_torso_flat_orient = torch.zeros(
                self.env.num_envs, device=self.env.device
            )  # dims 1

        return reward_torso_flat_orient

    def _reward_limits_action(self):
        """
        Reward for pass the action limits
        """
        out_of_limits = -(self.env.actions_untreated - self.env.clip_actions_min).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.env.actions_untreated - self.env.clip_actions_max).clip(
            min=0.0
        )  # upper limit

        error_limits_action = torch.abs(out_of_limits)
        error_limits_action = torch.sum(error_limits_action, dim=1)  # dims 2->1

        reward_limits_action = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_action * error_limits_action
        )

        return reward_limits_action

    def _reward_limits_dof_pos(self):
        """
        Reward for pass the dof position limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.env.num_actions))
        # ----------------------------------------------

        out_of_limits = -(
                self.env.dof_pos[:, related_indexes]
                - self.env.soft_dof_pos_limits[:, 0][related_indexes]
        ).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (
                self.env.dof_pos[:, related_indexes]
                - self.env.soft_dof_pos_limits[:, 1][related_indexes]
        ).clip(
            min=0.0
        )  # upper limit

        error_limits_dof_pos = torch.abs(out_of_limits)
        error_limits_dof_pos = torch.sum(error_limits_dof_pos, dim=1)  # dims 2->1

        reward_limits_dof_pos = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_dof_pos * error_limits_dof_pos
        )

        return reward_limits_dof_pos

    def _reward_limits_dof_vel(self):
        """
        Reward for pass the dof velocity limits
        """

        # ----------------------------------------------
        # get all controllable joints
        related_indexes = list(range(self.env.num_actions))
        # ----------------------------------------------

        error_limits_dof_vel = (
                torch.abs(self.env.dof_vel[:, related_indexes])
                - self.env.soft_dof_vel_limits[related_indexes]
        ).clip(min=0.0)
        error_limits_dof_vel = torch.sum(error_limits_dof_vel, dim=1)  # dims 2->1

        reward_limits_dof_vel = 1 - torch.exp(
            self.env.cfg.rewards.sigma_limits_dof_vel * error_limits_dof_vel
        )

        return reward_limits_dof_vel

    def _reward_stand_still_foot_distance(self):
        """
        Reward for tracking the foot distance match with the stand command,
        when the gait is in the stand state (two feet on the ground).
        """

        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        error_foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])
        error_foot_distance_y = torch.abs(
            error_foot_distance_y - self.env.cfg.rewards.stand_still_foot_distance
        )
        error_foot_distance_y = torch.sum(error_foot_distance_y, dim=1)  # dims 2->1

        reward_stand_still_foot_distance = torch.exp(
            self.env.cfg.rewards.sigma_stand_still_foot_distance * error_foot_distance_y
        )

        # ----------------------------
        # 一只脚当前正在触地，另一只脚刚刚触地
        condition_left_foot_contact = self.env.feet_contact[:, 0]  # dims 1
        condition_left_foot_contact_trig = self.env.feet_contact_trig[:, 0]  # dims 1
        condition_right_foot_contact = self.env.feet_contact[:, 1]  # dims 1
        condition_right_foot_contact_trig = self.env.feet_contact_trig[:, 1]  # dims 1
        condition_both_feet_contact = (
                condition_left_foot_contact * condition_right_foot_contact_trig
                + condition_left_foot_contact_trig * condition_right_foot_contact
        )

        # ----------------------------
        # 处于站立命令
        selector_stand = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        selector_stand[self.env.env_ids_of_stand_command] = 1

        # ----------------------------
        reward_stand_still_foot_distance *= condition_both_feet_contact
        reward_stand_still_foot_distance *= selector_stand

        return reward_stand_still_foot_distance

    def _reward_feet_distance_too_close(self):
        """
        Penalty for the distance between the feet being too close.
        """
        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        foot_distance_x = torch.abs(left_foot_pos[:, 0:1] - right_foot_pos[:, 0:1])
        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])
        foot_distance_z = torch.abs(left_foot_pos[:, 2:3] - right_foot_pos[:, 2:3])

        foot_distance = torch.sqrt(foot_distance_y ** 2)

        # st()
        error_foot_distance_too_close = torch.abs(
            foot_distance - self.env.cfg.rewards.feet_distance_too_close
        ) * (foot_distance < self.env.cfg.rewards.feet_distance_too_close)
        # print(foot_distance_y)
        # error_foot_distance_y_too_far = torch.abs(foot_distance_y - self.env.cfg.rewards.feet_distance_y_too_far) \
        #                                  * (foot_distance_y >self.env.cfg.rewards.feet_distance_y_too_far)

        error_foot_distance_too_close = torch.sum(error_foot_distance_too_close, dim=1)  # dims 2->1
        # error_foot_distance_too_far = torch.sum(error_foot_distance_y_too_far, dim=1)  # dims 2->1

        reward_feet_distance_too_close = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_distance_too_close * (error_foot_distance_too_close)
        )

        return reward_feet_distance_too_close

    def _reward_feet_distance_y_too_close(self):
        """
        Penalty for the distance between the feet in the y direction being too close.
        """
        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])

        error_foot_distance_y_too_close = torch.abs(
            foot_distance_y - self.env.cfg.rewards.feet_distance_y_too_close
        ) * (foot_distance_y < self.env.cfg.rewards.feet_distance_y_too_close)
        error_foot_distance_y_too_close = torch.sum(error_foot_distance_y_too_close, dim=1)

        reward_feet_distance_y_too_close = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_distance_y_too_close * error_foot_distance_y_too_close
        )

        return reward_feet_distance_y_too_close

    def _reward_feet_distance_too_far(self):
        """
        Penalty for the distance between the feet being too close.
        """
        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        foot_distance_x = torch.abs(left_foot_pos[:, 0:1] - right_foot_pos[:, 0:1])
        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])
        foot_distance_z = torch.abs(left_foot_pos[:, 2:3] - right_foot_pos[:, 2:3])

        foot_distance = torch.sqrt(foot_distance_y ** 2)

        # st()
        error_foot_distance_too_far = torch.abs(
            foot_distance - self.env.cfg.rewards.feet_distance_too_far
        ) * (foot_distance > self.env.cfg.rewards.feet_distance_too_far)
        # print(foot_distance_y)
        # error_foot_distance_y_too_far = torch.abs(foot_distance_y - self.env.cfg.rewards.feet_distance_y_too_far) \
        #                                  * (foot_distance_y >self.env.cfg.rewards.feet_distance_y_too_far)

        error_foot_distance_too_far = torch.sum(error_foot_distance_too_far, dim=1)  # dims 2->1
        # error_foot_distance_too_far = torch.sum(error_foot_distance_y_too_far, dim=1)  # dims 2->1

        reward_feet_distance_too_far = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_distance_too_far * (error_foot_distance_too_far)
        )

        return reward_feet_distance_too_far

    def _reward_feet_distance_y_too_far(self):
        """
        Penalty for the distance between the feet in the y direction being too far.
        """
        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        foot_distance_y = torch.abs(left_foot_pos[:, 1:2] - right_foot_pos[:, 1:2])

        error_foot_distance_y_too_far = (
                torch.abs(foot_distance_y - self.env.cfg.rewards.feet_distance_y_too_far)
                * (foot_distance_y > self.env.cfg.rewards.feet_distance_y_too_far)
                * (self.env.commands[:, 1].reshape(-1, 1) < self.env.cfg.rewards.velocity_threshold)
        )

        error_foot_distance_y_too_far = torch.sum(error_foot_distance_y_too_far, dim=1)

        reward_feet_distance_y_too_close = 1 - torch.exp(
            self.env.cfg.rewards.sigma_feet_distance_y_too_far * error_foot_distance_y_too_far
        )
        # st()
        # print(f'feet:{reward_feet_distance_y_too_close}')
        return reward_feet_distance_y_too_close

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.env.rigid_body_states[:, self.env.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.env.cfg.rewards.min_dist
        max_df = self.env.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
                torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.env.last_root_vel - self.env.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.env.commands[:, :2] - self.env.base_lin_vel[:, :2], dim=1
        )
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error


    def _reward_feet_distance_x_too_close(self):
        """
        Penalty for the distance between the feet in the y direction being too close.
        """
        left_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 0, 0:3
        ]
        right_foot_pos_in_world_frame = self.env.rigid_body_states[:, self.env.feet_indices][
            :, 1, 0:3
        ]

        left_foot_pos_to_base_in_world_frame = (
                left_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )
        right_foot_pos_to_base_in_world_frame = (
                right_foot_pos_in_world_frame - self.env.root_states[:, 0:3]
        )

        left_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame
        )
        right_foot_pos = quat_rotate_inverse(
            self.env.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame
        )

        foot_distance_x = torch.abs(left_foot_pos[:, 0:1] - right_foot_pos[:, 0:1])

        contact_sign = self.env.feet_contact[:, 0] & self.env.feet_contact[:, 1]

        error_foot_distance_x_too_close = torch.abs(
            self.env.cfg.rewards.feet_distance_x_too_close - foot_distance_x
        )
        error_foot_distance_x_too_close = (
                torch.sum(error_foot_distance_x_too_close, dim=1) * contact_sign
        )

        # print(error_foot_distance_x_too_close)

        reward_feet_distance_x_too_close = torch.where(
            contact_sign,
            torch.exp(
                self.env.cfg.rewards.sigma_feet_distance_x_too_close
                * error_foot_distance_x_too_close
            ),
            torch.tensor(0, device=error_foot_distance_x_too_close.device),
        )

        # print(reward_feet_distance_x_too_close)
        return reward_feet_distance_x_too_close

    def _neg_exp(self, x, a=1):
        """ shorthand helper for negative exponential e^(-x/a)
            a: range of x
        """
        return torch.exp(-(x / a) / a)

    def _neg_sqrd_exp(self, x, a=1):
        """ shorthand helper for negative squared exponential e^(-(x/a)^2)
            a: range of x
        """
        return torch.exp(-torch.square(x / a) / a)

