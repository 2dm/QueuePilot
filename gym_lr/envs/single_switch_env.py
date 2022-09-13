import itertools
import json
import time
from abc import ABC
import random
from typing import Type, Union

import ray
from ray.tune import register_env

from gym_lr.rl_src.configurations import MBPS_TO_PKTS, get_buffer_size, NUM_OF_CONNECTIONS, TRAINING_DELAYS, \
    TRAINING_NUM_OF_SENDERS, TEST_BUFFERS
from gym_lr.envs.env_base import EnvBase
from gym_lr.switch.const_switch import ConstSwitch
from gym_lr.switch.learning_switch import LearningSwitch
from gym_lr.switch.red_switch import REDSwitch
from gym_lr.switch.tail_drop_switch import TailDropSwitch
from utils import _fix_permissions, write_rec_to_file


class SingleSwitchEnv(EnvBase):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config, switch_class: Type[Union[LearningSwitch, REDSwitch, TailDropSwitch]], switch_config=dict({})):
        super().__init__(env_config)
        switch_config.update({
            "switch_name": "s1",
            "bw": self.topo_bw,
            "buf_size": self.topo_buf_size,
            "delay": self.topo_delay,
            "logs_episode_interval": self.logs_episode_interval,
            "log_dir": self.log_dir,
            "create_bpf": self.create_bpf,
            "egress_if": env_config["egress_if"],
            "ingress_if": env_config["ingress_if"],
            "num_of_steps": self.num_of_steps,
            "no_senders": env_config["no_senders"],
        })
        self.agent = switch_class(switch_config)

        self.observation_space = self.agent.observation_space
        self.action_space = self.agent.action_space

        self.env_parameters_stats = []

    def reset_obs(self):
        self.agent.reset_obs()

    def apply_actions(self, actions):
        self.agent.apply_action(actions)

    def finalize_step(self):
        obs, reward, done, info = self.agent.finalize_step()
        self.total_reward += reward
        return obs, reward, done, info

    def reset_agents(self):
        return self.agent.reset()

    def close(self):
        self.agent.close()
        self.real_net.close()

    def get_tables(self):
        return self.agent.get_tables()

    def set_tables(self, new_tables):
        self.agent.set_tables(new_tables)

    def change_topo(self, new_topo):
        self.topo_delay = new_topo["delay"]
        self.topo_buf_size = new_topo["buf_size"]
        self.topo_bw = new_topo["bw"]
        self.no_senders = new_topo["no_senders"]
        return self.agent.change_topo(new_topo)

    def get_topo_params(self):
        return {"buffer": self.topo_buf_size, "delay": self.topo_delay, "senders": self.no_senders}

    def get_reward(self):
        return self.total_reward


    def _apply_new_topo_params(self, bw, delay, no_sdr, buffer=None):
        self.real_net.start_qdiscs(delay=delay)
        buf_size = get_buffer_size(bw=bw, delay=delay, senders=no_sdr) if buffer is None else buffer
        topo_params = {
            "delay": delay,
            "buf_size": buf_size,
            "bw": bw,
            "no_senders": no_sdr,
        }

        self.real_net.change_topo_params(topo_params)
        self.env_parameters_stats.append({
            'Episode': self.current_episode,
            'Buffer size': buf_size,
            'BW': bw,
            'Delay': delay,
            'Senders': no_sdr,
        })
        self.episode_records.append(topo_params)
        ret = self.change_topo(topo_params)

        if ret != 0:
            self.real_net.kill_senders()

    def change_topo_rand(self):
        delays = TRAINING_DELAYS
        senders = TRAINING_NUM_OF_SENDERS

        delay = random.choice(delays)
        no_sdr = random.choice(senders)
        bw = 1000

        self._apply_new_topo_params(bw, delay, no_sdr)

        bg_traffic_mb = 10 * random.choice(range(1, 11))
        self.real_net.start_senders_iperfs(no_senders=no_sdr, bg_traffic_mb=bg_traffic_mb)

    def change_buffer_rand(self):
        delay = TRAINING_DELAYS[0]
        senders = TRAINING_NUM_OF_SENDERS[0]
        buffers = TEST_BUFFERS

        buffer = random.choice(buffers)
        bw = 1000

        self._apply_new_topo_params(bw, delay, senders, buffer)

        bg_traffic_mb = 10 * random.choice(range(1, 11))
        self.real_net.start_senders_iperfs(no_senders=senders, bg_traffic_mb=bg_traffic_mb)

    def start_episode_flows(self):
        senders = TRAINING_NUM_OF_SENDERS[0]
        bg_traffic_mb = 10 * random.choice(range(1, 11))
        self.real_net.start_senders_iperfs(no_senders=senders, bg_traffic_mb=bg_traffic_mb)

    def log_training_envs(self):
        log_filename = f"{self.log_dir}/training_envs.json"
        write_rec_to_file(self.env_parameters_stats, log_filename)

    def env_kill_senders(self):
        self.real_net.kill_senders()
        if self.current_episode % 100 == 0:
            self.real_net.kill_servers()
            self.real_net.start_server_iperfs(no_senders=max(TRAINING_NUM_OF_SENDERS))

    def change_log_dir(self, new_label):
        self.label = new_label
        self.log_dir = f"logs/{new_label}/"
        self.agent.change_log_dir(self.log_dir)

    def reset_episodes(self):
        self.current_episode = 0
        self.log_this_episode = False
        self.agent.reset_episodes()


class SingleLearningSwitchEnv(SingleSwitchEnv):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config):
        super().__init__(env_config, LearningSwitch)


class SingleREDSwitchEnv(SingleSwitchEnv):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config):
        super().__init__(env_config, REDSwitch,
                         switch_config=dict({
                             "red_min_threshold": env_config["min"],
                             "red_max_threshold": env_config["max"],
                             "limit": env_config["limit"],
                             "red_max_p": env_config["red_max_p"],
                             "burst": env_config["burst"]
                         }))

    def step(self, actions=None):
        return super().step(actions)


class SingleTailDropSwitchEnv(SingleSwitchEnv):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config):
        super().__init__(env_config, TailDropSwitch)

    def step(self, actions=None):
        return super().step(actions)


class SingleConstSwitchEnv(SingleSwitchEnv):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config):
        super().__init__(env_config, ConstSwitch, switch_config=dict({
            "const_threshold_ratio": env_config["const_threshold_ratio"]
        }))
