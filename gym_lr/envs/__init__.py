from ray.tune import register_env
from ray.rllib.models import ModelCatalog

from gym_lr.rl_src.my_torch_rnn_model import MyTorchRNNModel
from gym_lr.envs.single_switch_env import SingleLearningSwitchEnv, SingleREDSwitchEnv, SingleConstSwitchEnv, \
    SingleTailDropSwitchEnv

register_env("single_learning_switch_env", lambda env_config: SingleLearningSwitchEnv(env_config))
register_env("single_red_switch_env", lambda env_config: SingleREDSwitchEnv(env_config))
register_env("single_const_switch_env", lambda env_config: SingleConstSwitchEnv(env_config))
register_env("single_tail_drop_switch_env", lambda env_config: SingleTailDropSwitchEnv(env_config))

ModelCatalog.register_custom_model("my_rnn", MyTorchRNNModel)
