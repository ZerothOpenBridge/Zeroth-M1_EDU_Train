from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

# ------------------------------------------------------------
# Base
import legged_gym.envs.base


from legged_gym.envs.mini1.mini_code import MINI1_APPEARANCE

from legged_gym.envs.mini1.mini_18dof_walk_config import (
    walk_18dof_MINI1MainBodyCfg,
    walk_18dof_MINI1MainBodyCfgPPO,
)

task_registry.register(
    "mini1_18dof_walk",
    MINI1_APPEARANCE,
    walk_18dof_MINI1MainBodyCfg(),
    walk_18dof_MINI1MainBodyCfgPPO(),
)
