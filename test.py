import datetime
import os
import random
import time
import numpy as np
import ray
from ray.rllib.agents import ppo

from gym_lr.rl_src.configurations import SINGLE_SWITCH_EVAL_TOPO_PARAMS, NUM_OF_EPISODES_TO_TEST, \
    SINGLE_SWITCH_TEST_CONFIG, LSTM_SIZE
from gym_lr.envs.realnet_networks import RealPfifoNet, EGRESS_IF, INGRESS_IF
from utils import create_log_dir, write_params_log

os.nice(-20)


def build_setup(label, topo_params, agent_checkpoint, seed=None):
    log_dir = f"logs/{label}"
    create_log_dir(log_dir)
    print(label)

    config = SINGLE_SWITCH_TEST_CONFIG
    config['env_config']['label'] = label

    config['env_config'].update(topo_params)
    config['env_config'].update({'egress_if': EGRESS_IF,
                                 'ingress_if': INGRESS_IF})

    seed = int(random.uniform(0, 2 ** 32 - 1)) if (seed is None) else seed
    config['seed'] = seed
    random.seed(seed)

    real_net = RealPfifoNet(qdisc_params=topo_params)
    config['env_config'].update({'real_net': real_net})

    config["env_config"]["create_bpf"] = False
    agent = ppo.PPOTrainer(config=config, env="single_learning_switch_env")
    agent.load_checkpoint(agent_checkpoint)

    config["env_config"]["create_bpf"] = True
    env = agent.env_creator(config["env_config"])

    write_params_log(log_dir=log_dir, algorithm="RL",
                     checkpoint=agent_checkpoint,
                     first_topo_param=topo_params,
                     label=label)

    return real_net, agent, env


def change_setup_params(label, real_net, agent, env, new_topo_params):
    real_net.change_topo_params(new_topo_params)
    agent.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.change_topo(new_topo_params)))
    env.change_topo(new_topo_params)
    create_log_dir(f"logs/{label}")
    env.change_log_dir(label)


def test_single_env(topo_params, real_net, agent, env, start_flows_func):
    real_net.kill_connections()
    real_net.set_next_router_qdisc(max_buf=10000)
    real_net.start_qdiscs(delay=topo_params["delay"])
    real_net.start_server_iperfs(no_senders=topo_params["no_senders"])

    for ii in range(NUM_OF_EPISODES_TO_TEST):
        episode_reward = 0
        done = False
        action = 0
        reward = 0
        state = [np.zeros(LSTM_SIZE, np.float32), np.zeros(LSTM_SIZE, np.float32)]

        obs = env.reset()

        start_flows_func(no_senders=topo_params["no_senders"])

        while not done:
            action, state, logits = agent.compute_single_action(obs, state, prev_action=action, prev_reward=reward)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        real_net.kill_senders()
        real_net.kill_tshark()
    env.reset()


def test_single_fct_env(topo_params, real_net, agent, env, msg_size, label):
    real_net.kill_connections()
    real_net.set_next_router_qdisc(max_buf=10000)
    real_net.start_qdiscs(delay=topo_params["delay"])
    real_net.start_server_iperfs(no_senders=topo_params["no_senders"])

    for ii in range(NUM_OF_EPISODES_TO_TEST):
        episode_reward = 0
        done = False
        action = 0
        reward = 0
        state = [np.zeros(LSTM_SIZE, np.float32), np.zeros(LSTM_SIZE, np.float32)]

        obs = env.reset()

        real_net.start_fct_senders_iperfs(no_senders=topo_params["no_senders"],
                                          tshark_log=label, ep=ii, msg_size=f"{msg_size}")

        while not done:
            action, state, logits = agent.compute_single_action(obs, state, prev_action=action, prev_reward=reward)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        real_net.kill_senders()
        real_net.kill_tshark()
    env.reset()


def close_all(env, real_net, agent):
    real_net.kill_connections()
    real_net.kill_tshark()
    agent.stop()
    env.close()


def run():
    label = "test_"
    checkpoint = ""
    initial_topo_params = SINGLE_SWITCH_EVAL_TOPO_PARAMS

    start = time.time()
    print(f'Starting test at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")}')
    ray.init()
    real_net, agent, env = build_setup(label=label, topo_params=initial_topo_params, agent_checkpoint=checkpoint)
    test_single_env(initial_topo_params, real_net, agent, env, real_net.start_senders_iperfs)
    close_all(env, real_net, agent)

    ray.shutdown()
    print(f'Completed at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")} ; '
          f'{str(datetime.timedelta(seconds=time.time() - start))}')


def run_auto_tune():
    label = "test_automatic_tuning_"
    checkpoint = ""
    initial_topo_params = SINGLE_SWITCH_EVAL_TOPO_PARAMS

    start = time.time()
    print(f'Starting test at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")}')
    ray.init()
    real_net, agent, env = build_setup(label=label, topo_params=initial_topo_params, agent_checkpoint=checkpoint)
    test_single_env(initial_topo_params, real_net, agent, env, real_net.start_changing_load_senders_iperfs)
    close_all(env, real_net, agent)

    ray.shutdown()
    print(f'Completed at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")} ; '
          f'{str(datetime.timedelta(seconds=time.time() - start))}')


def run_fct():
    base_label = "test_FCT_"
    checkpoint = ""
    initial_topo_params = SINGLE_SWITCH_EVAL_TOPO_PARAMS

    start = time.time()
    print(f'Starting FCT test at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")}')
    ray.init()
    for msg_size in ["1k", "10k", "100k", "1m", "10m"]:
        label = f"{base_label}_{msg_size}"
        real_net, agent, env = build_setup(label=label, topo_params=initial_topo_params, agent_checkpoint=checkpoint)
        test_single_fct_env(initial_topo_params, real_net, agent, env, msg_size=msg_size, label=label)
        close_all(env, real_net, agent)

    ray.shutdown()
    print(f'Completed at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")} ; '
          f'{str(datetime.timedelta(seconds=time.time() - start))}')


if __name__ == '__main__':
    run()
    # run_auto_tune()
    # run_fct()
