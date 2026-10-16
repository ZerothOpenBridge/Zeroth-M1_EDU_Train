from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from pdb import set_trace as st
import copy
from pynput import keyboard
import time
from threading import Thread, Event
import matplotlib.pyplot as plt

import socket
import json


class cmd:
    vx = 0.03
    vy = 0.0
    dyaw = 0
    # render_v = 0


def on_press(key):
    """监听按键按下事件"""
    try:
        if key.char == "8":  # 数字键盘 8 - 前进
            cmd.vx += 0.01
        elif key.char == "2":  # 数字键盘 5 - 后退
            cmd.vx -= 0.01
        elif key.char == "4":  # 数字键盘 4 - 左移
            cmd.vy += 0.01
        elif key.char == "6":  # 数字键盘 6 - 右移
            cmd.vy -= 0.01
        elif key.char == "7" or key.char == "1":  # 数字键盘 7 / 1- 左转
            cmd.dyaw += 0.1
        elif key.char == "9" or key.char == "3":  # 数字键盘 9 / 3- 右转
            cmd.dyaw -= 0.1
        elif key.char == "0":
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.dyaw = 0.0
        elif key.char == "x":
            cmd.render_v += 5
        print("command: ", cmd.vx, cmd.vy, cmd.dyaw)

    except AttributeError:
        pass


def on_release(key):
    """监听按键释放事件"""
    if key == keyboard.Key.esc:  # 按下 ESC 键退出程序
        return False


# EXPORT_POLICY=True
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


class PlotData:
    def __init__(self):
        self.count_lowlevel = 0
        self.dq = np.zeros(12)
        self.q = np.zeros(12)
        self.tau = np.zeros(12)
        self.hip_pitch_pos = np.zeros(2)
        self.hip_torque = np.zeros(2)
        self.projected_gravity = np.zeros(3)
        self.stop_event = Event()
        self.action = np.zeros(12)


def plot_motor_data(self):
    plt.ion()  # Enable interactive mode

    # Create three figures
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10))  # For velocities
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10))  # For positions
    fig3, axes3 = plt.subplots(3, 2, figsize=(12, 10))  # For actions
    fig1.suptitle("Motor Velocities", fontsize=16)
    fig2.suptitle("Motor Positions", fontsize=16)
    fig3.suptitle("Motor Actions", fontsize=16)

    t = []
    joint_velocities = [[] for _ in range(12)]  # Store velocities for all 12 joints
    joint_positions = [[] for _ in range(12)]  # Store positions for all 12 joints
    joint_actions = [[] for _ in range(12)]  # Store actions for all 12 joints
    velocity_lines = []
    position_lines = []
    action_lines = []

    # Initialize plots for velocities
    for i, ax in enumerate(axes1.flat):
        (line1,) = ax.plot([], [], "r-", label=f"Joint {i}")  # Red line for i
        (line2,) = ax.plot([], [], "b-", label=f"Joint {i+6}")  # Blue line for i+6
        velocity_lines.append((line1, line2))
        ax.set_title(f"Joint Velocity {i},{i+6}")
        ax.legend()

    # Initialize plots for positions
    for i, ax in enumerate(axes2.flat):
        (line1,) = ax.plot([], [], "r-", label=f"Joint {i}")  # Red line for i
        (line2,) = ax.plot([], [], "b-", label=f"Joint {i+6}")  # Blue line for i+6
        position_lines.append((line1, line2))
        ax.set_title(f"Joint Position {i},{i+6}")
        ax.legend()

    # Initialize plots for actions
    for i, ax in enumerate(axes3.flat):
        (line1,) = ax.plot([], [], "r-", label=f"Joint {i}")  # Red line for i
        (line2,) = ax.plot([], [], "b-", label=f"Joint {i+6}")  # Blue line for i+6
        action_lines.append((line1, line2))
        ax.set_title(f"Joint Action {i},{i+6}")
        ax.legend()

    while not self.stop_event.is_set():
        t.append(self.count_lowlevel * 0.001)  # Time in seconds

        # Update velocity data
        for i in range(12):
            joint_velocities[i].append(self.dq[i])
            joint_positions[i].append(self.q[i])
            joint_actions[i].append(self.action[i])  # Add action data
        # print(joint_velocities)
        # Update velocity plots
        for i, (line1, line2) in enumerate(velocity_lines):
            line1.set_xdata(t)
            line1.set_ydata(joint_velocities[i])
            line2.set_xdata(t)
            line2.set_ydata(joint_velocities[i + 6])
            axes1.flat[i].relim()
            axes1.flat[i].autoscale_view()

        # Update position plots
        for i, (line1, line2) in enumerate(position_lines):
            line1.set_xdata(t)
            line1.set_ydata(joint_positions[i])
            line2.set_xdata(t)
            line2.set_ydata(joint_positions[i + 6])
            axes2.flat[i].relim()
            axes2.flat[i].autoscale_view()

        # Update action plots
        for i, (line1, line2) in enumerate(action_lines):
            line1.set_xdata(t)
            line1.set_ydata(joint_actions[i])
            line2.set_xdata(t)
            line2.set_ydata(joint_actions[i + 6])
            axes3.flat[i].relim()
            axes3.flat[i].autoscale_view()

        plt.draw()
        plt.pause(2)

    plt.ioff()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


plot_data = PlotData()


# motor_data_thread = Thread(target=plot_motor_data, args=(plot_data,))
# motor_data_thread.start()
def export_policy_to_onnx(actor_critic, path, env_cfg):

    # st()
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "policy_onnx.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()  # 设置模型为评估模式

    # 创建示例输入
    # 您需要根据模型的实际输入形状进行修改
    batch_size = 1  # 可以根据需要调整
    # st()
    num_observations = env_cfg.env.num_obs * env_cfg.env.num_stack  # frame_stack * num_single_obs
    dummy_input = torch.randn(batch_size, num_observations)

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,  # 模型
        dummy_input,  # 示例输入
        model_path,  # 导出路径
        export_params=True,  # 导出模型参数
        opset_version=11,  # ONNX opset 版本，可以根据需要调整
        do_constant_folding=True,  # 进行常量折叠优化
        input_names=["input"],  # 输入节点名称，可以根据需要调整
        output_names=["output"],  # 输出节点名称，可以根据需要调整
    )
    print(f"模型已导出为 ONNX 格式，保存路径：{model_path}")


def compute_env_obsbuf(env):
    sin_phase = torch.sin(2 * torch.pi * env.phase).unsqueeze(-1)
    cos_phase = torch.cos(2 * torch.pi * env.phase).unsqueeze(-1)
    # st()
    obs_buf = torch.cat(
        (
            # command
            sin_phase,
            cos_phase,
            env.commands[:, 0:3]
            * torch.tensor([2, 2, 1]).to(env.device),  # 2m/s in xy, 1 rad/s yaw
            # base related
            # self.base_ang_vel * self.obs_scales.ang_vel,
            # self.base_projected_gravity * self.obs_scales.gravity,
            # dof related
            env.dof_pos_offset * 1,
            env.dof_vel * 0.05,
            # action related
            env.actions * 1,
            env.base_ang_vel * 1,
            env.base_quat * 1,
        ),
        dim=-1,
    )

    return obs_buf


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # st()
    # override some parameters for testing
    env_cfg.env.episode_length_s = 600.0
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

    #env_cfg.terrain.num_rows = 5
    #env_cfg.terrain.num_cols = 5
    #env_cfg.terrain.curriculum = False

    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env.commands[:,0]=0.2
    obs = env.get_observations()
    # st()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
        )
        # path = export_policy_as_jit(ppo_runner.algorithm.actor_critic, path)
        export_policy_as_jit(ppo_runner.algorithm.actor_critic, path)
        export_policy_to_onnx(ppo_runner.algorithm.actor_critic, path, env_cfg)
        print(f"\033[93m" f"EXPORT_POLICY: " f"Exported policy as jit script to: " f"{path}")

    # -------------------------------------------------------
    obs_list = []
    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 200  # number of steps before plotting states
    print_shoulder_interval = 50  # 每 N 步打印一次肩 pitch，避免刷屏
    print_foot_height_interval = 50  # ~*~每 N 步打印一次左脚高度分解
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    start = 0
    # env.commands[:,0]=0.05s
    env.commands[:, 0] = 0.03
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    # logger.plot_states()
    actions = torch.zeros([1, env.num_actions], device=env.device)
    for i in range(10 * int(env.max_episode_length)):
        env.commands[:,] = torch.tensor([cmd.vx, cmd.vy, cmd.dyaw], dtype=torch.float32).to(
            env.device
        )

        actions = policy(obs.detach())

        """
        obs_reshaped = obs.cpu().numpy().reshape(1, 1, 1, 225)
        if len(obs_list) < 100:
            obs_list.append(obs_reshaped)
        elif len(obs_list) == 100:
            obs_array = np.stacsk(obs_list, axis=0)
            #np.save("obs_100.npy", obs_array)
        # print("已保存100组obs到 obs_100.npy")
        """
        obs, _, rews, dones, infos = env.step(actions.detach())
        if (
            i % print_shoulder_interval == 0
            and hasattr(env, "shoulder_pitch_indices")
            and len(env.shoulder_pitch_indices) >= 2
        ):
            li, ri = env.shoulder_pitch_indices[0], env.shoulder_pitch_indices[1]
            l_pos = env.dof_pos[robot_index, li].item()
            r_pos = env.dof_pos[robot_index, ri].item()
            print(
                f"shoulder_pitch dof_pos [L,R] rad: {l_pos:.4f}, {r_pos:.4f}  "
                f"sum={l_pos + r_pos:.4f}  cmd=({cmd.vx:.2f},{cmd.vy:.2f},{cmd.dyaw:.2f})"
            )
        #~*~ 打印脚高度
        if i % print_foot_height_interval == 0 and len(env.feet_indices) >= 1:
            left_foot_link_z = env.rigid_body_states[robot_index, env.feet_indices[0], 2].item()
            left_foot_height = env.get_left_foot_height()[robot_index, 0].item()
            foot_bottom_z = left_foot_link_z - env.cfg.asset.foot_thickness

            if isinstance(env.measured_heights, torch.Tensor):
                measured_heights_mean = env.measured_heights[robot_index].mean().item()
                measured_heights_min = env.measured_heights[robot_index].min().item()
                measured_heights_max = env.measured_heights[robot_index].max().item()
            else:
                measured_heights_mean = float(env.measured_heights)
                measured_heights_min = float(env.measured_heights)
                measured_heights_max = float(env.measured_heights)

            print(
                "left_foot_height debug: "
                f"link_z={left_foot_link_z:.4f}, "
                f"foot_thickness={env.cfg.asset.foot_thickness:.4f}, "
                f"link_z-thickness={foot_bottom_z:.4f}, "
                f"measured_h(mean/min/max)={measured_heights_mean:.4f}/"
                f"{measured_heights_min:.4f}/{measured_heights_max:.4f}, "
                f"get_left_foot_height={left_foot_height:.4f}"
            )
        # st()

        plot_data.dq = env.dof_vel[robot_index].detach().cpu().numpy().reshape(-1)  # 存储关节速度数据
        plot_data.q = env.dof_pos_offset[robot_index].detach().cpu().numpy().reshape(-1)  # 存储关节位置数据
        plot_data.action = env.actions[robot_index].detach().cpu().numpy().reshape(-1)  # 存储关节动作数据
        base_projected_gravity = env.base_projected_gravity[robot_index].detach().cpu().numpy().reshape(-1)  # 存储基座重力投影数据
        base_ang_vel = env.base_ang_vel[robot_index].detach().cpu().numpy().reshape(-1)  # 存储基座角速度数据
        dof_tor_ss = env.dof_tor[robot_index].detach().cpu().numpy().reshape(-1)  # 存储关节力矩数据

        # plot_data.count_lowlevel = i
        # plot_data.projected_gravity = np.array(env.base_projected_gravity.to("cpu")).reshape(3)
        # st()

        if RECORD_FRAMES:  # 0
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        if MOVE_CAMERA:  # 0
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        # st()
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2]
                    .cpu()
                    .numpy(),
                }
            )

        elif i == stop_state_log:  # 100
            logger.plot_states()

        if 0 < i < stop_rew_log:  # 30000s
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    # plot_data.stop_event.set()
    # velocity_thread.join()
    # projected_gravity_thread.join()
    # pos_thread.join()
    # motor_data_thread.join()


if __name__ == "__main__":


    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
