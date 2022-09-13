import os
import random
from ray.rllib.agents import ppo
import ray
from ray.rllib.models import ModelCatalog

from gym_lr.rl_src.my_torch_rnn_model import MyTorchRNNModel
from gym_lr.rl_src.configurations import SINGLE_SWITCH_CONFIG, NUM_OF_EPISODES_TO_TRAIN, SINGLE_SWITCH_EVAL_TOPO_PARAMS, \
    TRAINING_NUM_OF_SENDERS
from gym_lr.envs.realnet_networks import RealPfifoNet, EGRESS_IF, INGRESS_IF
from utils import create_log_dir, write_params_log

os.nice(-20)


def train(label, checkpoint=None):
    log_dir = f"logs/{label}"
    create_log_dir(log_dir)
    print(label)

    config = SINGLE_SWITCH_CONFIG
    config['env_config']['label'] = label

    initial_topo_params = SINGLE_SWITCH_EVAL_TOPO_PARAMS
    config['env_config'].update(initial_topo_params)
    config['env_config'].update({'egress_if': EGRESS_IF,
                                 'ingress_if': INGRESS_IF})

    seed = int(random.uniform(0, 2 ** 32 - 1))
    config['seed'] = seed

    real_net = RealPfifoNet(qdisc_params=initial_topo_params)
    config['env_config'].update({'real_net': real_net})

    ppo_trainer = ppo.PPOTrainer(config=config, env="single_learning_switch_env")
    if checkpoint is not None:
        ppo_trainer.restore(checkpoint)

    random.seed(seed)

    write_params_log(log_dir=log_dir, algorithm="RL",
                     checkpoint=checkpoint,
                     first_topo_param=initial_topo_params,
                     label=label)

    real_net.kill_connections()
    real_net.start_qdiscs(delay=initial_topo_params['delay'])
    real_net.start_server_iperfs(no_senders=max(TRAINING_NUM_OF_SENDERS))
    real_net.set_next_router_qdisc(max_buf=10000)

    for ii in range(NUM_OF_EPISODES_TO_TRAIN):

        result = ppo_trainer.train()

        if ii != 0 and ii % 10 == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)

    checkpoint = ppo_trainer.save()
    print(f"checkpoint saved at {checkpoint}")
    real_net.kill_connections()

    ppo_trainer.workers.local_worker().foreach_env(lambda w: w.log_training_envs())

    return checkpoint


if __name__ == '__main__':
    ray.init()
    label = "train_"
    train(label=label)
    ray.shutdown()
