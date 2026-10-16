import os
import sys
import numpy as np
import isaacgym
import torch

from datetime import datetime
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


# from pycallgraph2 import PyCallGraph
# from pycallgraph2.output import GraphvizOutput
# from pycallgraph2 import Config


def train(args):
    # 创建环境
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # 创建ppo算法和训练参数
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # 训练开始
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    DEBUG_USE_FIXED_ARGS = False  # 调试完改回 False

    if DEBUG_USE_FIXED_ARGS:
        sys.argv = [
            sys.argv[0],
            "--task=mini1_12dof",
            "--num_envs=4096",
            "--headless",
        ]

    print(f"[DEBUG] sys.argv = {sys.argv}")
    args = get_args()
    print(f"[DEBUG] parsed args = {vars(args)}")

    train(args)
