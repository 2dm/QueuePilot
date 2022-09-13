import copy
from math import ceil, floor, sqrt
from typing import Dict

from ray.rllib import BaseEnv
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.utils.typing import PolicyID


EPISODE_RUN_TIME_IN_SEC = 5
EPISODE_RUN_TIME_IN_SEC_TEST = 10
SEC_PER_STEP = 0.005
STEPS_PER_EPISODE = int(EPISODE_RUN_TIME_IN_SEC / SEC_PER_STEP)
STEPS_PER_TEST_EPISODE = int(EPISODE_RUN_TIME_IN_SEC_TEST / SEC_PER_STEP)
NUM_EPISODES_COLLECT_BEFORE_SGD = 2
MBPS_TO_BITS = 1e6
PACKET_SIZE_BYTES = 1514
PACKET_SIZE_BYTES_ON_WIRE = 1538
BITS_TO_BYTES = 8
MBPS_TO_PKTS = MBPS_TO_BITS / (PACKET_SIZE_BYTES_ON_WIRE * BITS_TO_BYTES)
GBPS_TO_PKTS_PER_STEP = floor(1000 * MBPS_TO_PKTS * SEC_PER_STEP)
GBPS = 1e3 * MBPS_TO_BITS / BITS_TO_BYTES
SLEEP_CORRECTION = 0.00006
NUM_OF_EPISODES_TO_TRAIN = 10001 // NUM_EPISODES_COLLECT_BEFORE_SGD
NUM_OF_EPISODES_TO_TEST = 10

LOG_AVG_Q_WINDOW = 9
AVG_Q_WINDOW = 2 ** LOG_AVG_Q_WINDOW

TRAINING_DELAYS = [10, 20, 30, 40]
TEST_DELAYS = [10, 20, 30, 40, 50]
TRAINING_NUM_OF_SENDERS = [100, 200, 250]
TEST_NUM_OF_SENDERS = [100, 200, 300]
TEST_BUFFERS = None
NUM_OF_CONNECTIONS = 300
UDP_BITRATE = 80
LSTM_SIZE = 32

def get_buffer_size(bw, delay, senders):
    return ceil(floor(bw * MBPS_TO_PKTS) * 2 * (delay / 1000) / sqrt(senders))
    # return ceil(floor(bw * MBPS_TO_PKTS) * 2 * (delay / 1000))


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: Episode,
                         **kwargs) -> None:
        # print(f"{__name__} {time.time()}")
        base_env.vector_env.envs[0].change_topo_rand()
        # base_env.vector_env.envs[0].change_buffer_rand()
        # base_env.vector_env.envs[0].start_episode_flows()

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: Episode,
                       **kwargs):
        base_env.vector_env.envs[0].env_kill_senders()
        # print(f"{inspect.stack()[0][3]} {time.time()}")
        topo_params = base_env.vector_env.envs[0].get_topo_params()
        reward = base_env.vector_env.envs[0].get_reward()
        episode.custom_metrics[f"b_{topo_params['buffer']}-d_{topo_params['delay']}-s_{topo_params['senders']}"] = reward


MY_RNN_MODEL_CONFIG = {
    "custom_model": "my_rnn",
    "max_seq_len": 64,
    "lstm_use_prev_action": True,
    "lstm_use_prev_reward": True,
    "vf_share_layers": True,
}

SINGLE_SWITCH_CONFIG = {
    "num_workers": 0,
    "env_config": {
        "label": None,
        "logs_episode_interval": 100,
        "bw": 0,
        "buf_size": 0,
        "delay": 0,
        "create_bpf": True,
        "egress_if": None,
        "ingress_if": None,
        "real_net": None,
        "num_of_steps": STEPS_PER_EPISODE,
    },
    "model": copy.deepcopy(MY_RNN_MODEL_CONFIG),
    "framework": "torch",
    "train_batch_size": NUM_EPISODES_COLLECT_BEFORE_SGD * STEPS_PER_EPISODE,
    "sgd_minibatch_size": 256,
    "rollout_fragment_length": NUM_EPISODES_COLLECT_BEFORE_SGD * STEPS_PER_EPISODE,
    "batch_mode": "complete_episodes",
    "lr": 1e-4,
    "lr_schedule": [[0, 1e-4], [2000000, 1e-5], [5000000, 1e-5], [7000000, 1e-6]],
    "lambda": 0.97,
    "gamma": 0.995,
    "clip_param": 0.08,
    "entropy_coeff": 1e-5,
    "entropy_coeff_schedule": [[0, 1e-5], [1000000, 1e-6]],
    "vf_loss_coeff": 1e-4,
    "vf_clip_param": float("inf"),
    "seed": None,
    "callbacks": MyCallbacks,
    #  Evaluation
    "evaluation_interval": None,
    "evaluation_duration": 1,
    "custom_eval_function": None,
    "evaluation_config": {
        "env_config": {
            "delay": 0,
            "buf_size": 0,
            "bw": 0,
            "create_bpf": False,
        },
    }
}

SINGLE_SWITCH_TEST_CONFIG = copy.deepcopy(SINGLE_SWITCH_CONFIG)
SINGLE_SWITCH_TEST_CONFIG.update({
    "env_config": {
        "label": None,
        "logs_episode_interval": 1,
        "bw": 0,
        "buf_size": 0,
        "delay": 0,
        "create_bpf": False,
        "egress_if": None,
        "ingress_if": None,
        "num_of_steps": STEPS_PER_TEST_EPISODE,
    },
    "in_evaluation": True,
})

SINGLE_RED_SWITCH_TEST_CONFIG = copy.deepcopy(SINGLE_SWITCH_TEST_CONFIG)
SINGLE_RED_SWITCH_TEST_CONFIG["env_config"].update({
    "limit": 0,
    "min": 0,
    "max": 0,
    "burst": 0,
})

SINGLE_COSNT_SWITCH_TEST_CONFIG = copy.deepcopy(SINGLE_SWITCH_TEST_CONFIG)
SINGLE_COSNT_SWITCH_TEST_CONFIG["env_config"].update({"const_threshold": 0})

SINGLE_SWITCH_EVAL_TOPO_PARAMS = {
    "delay": 20,
    "buf_size": get_buffer_size(1000, 20, 200),
    "bw": 1000,
    "no_senders": NUM_OF_CONNECTIONS,
}
SINGLE_RED_SWITCH_EVAL_TOPO_PARAMS = copy.deepcopy(SINGLE_SWITCH_EVAL_TOPO_PARAMS)
SINGLE_RED_SWITCH_EVAL_TOPO_PARAMS.update({
    "limit": SINGLE_SWITCH_EVAL_TOPO_PARAMS["buf_size"] * PACKET_SIZE_BYTES,
    "min": SINGLE_SWITCH_EVAL_TOPO_PARAMS["buf_size"] * PACKET_SIZE_BYTES * 0.2,
    "max": SINGLE_SWITCH_EVAL_TOPO_PARAMS["buf_size"] * PACKET_SIZE_BYTES * 0.6,
    "burst": 1 + ceil(SINGLE_SWITCH_EVAL_TOPO_PARAMS["buf_size"] * 0.2),
    "red_max_p": 0.1,
})

