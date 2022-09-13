import abc
# import logging
import random
import time
from statistics import mean, stdev
import gym
import numpy as np
from gym_lr.rl_src.configurations import SEC_PER_STEP, SLEEP_CORRECTION
from plot_graphs_multiswitch import plot_episode_graph


class EnvBase(gym.Env):
    """A router simulator environment for OpenAI gym"""

    def __init__(self, env_config):
        super().__init__()

        self.label = env_config["label"]
        self.logs_episode_interval = env_config["logs_episode_interval"]
        self.topo_delay = env_config["delay"]
        self.topo_buf_size = env_config["buf_size"]
        self.topo_bw = env_config["bw"]
        self.log_dir = f"logs/{self.label}/"
        self.create_bpf = env_config["create_bpf"]
        self.real_net = env_config["real_net"]
        self.num_of_steps = env_config["num_of_steps"]
        self.no_senders = env_config["no_senders"]

        # gym and RL
        self.current_step = 0
        self.current_episode = 0
        self.total_reward = 0
        self.observation_space = None
        self.action_space = None

        # Operational
        # self.logger = logging.getLogger('event_log')
        # self.logger_format = logging.Formatter('{%(created)f %(message)s}')
        # self.logger.setLevel(logging.DEBUG)
        # self.event = StructuredMessage

        self.prev_step_start = 0
        self.prev_sleep_time = 0
        self.intervals_ms = []
        self.episode_records = []
        self.episodes_from_last_log = 0
        self.log_this_episode = False

    @abc.abstractmethod
    def reset_obs(self):
        """ TODO: add documentation"""

    @abc.abstractmethod
    def apply_actions(self, actions):
        """ TODO: add documentation"""

    @abc.abstractmethod
    def finalize_step(self):
        """ TODO: add documentation"""

    @abc.abstractmethod
    def reset_agents(self):
        """ TODO: add documentation"""

    def step(self, actions):
        step_start = time.time()
        self.current_step += 1

        # self.logger.debug(self.event(event=LrEvents.STEP_START, step=self.current_step))

        if self.current_step == 1:
            self.reset_obs()
        else:
            delta_ms = 1000 * (step_start - self.prev_step_start)
            self.intervals_ms.append(delta_ms)
        self.prev_step_start = step_start

        # self.executor.submit(self.agent.step, action)
        self.apply_actions(actions)

        # Wait step time
        self.time_interval()

        # Sample step observations and reward
        obs, reward, done, info = self.finalize_step()

        if done and self.log_this_episode:
            plot_episode_graph(log_dir=self.label, episode=self.current_episode, output_dir=self.log_dir)

        # self.logger.debug(self.event(event=LrEvents.STEP_COMPLETE, step=self.current_step))

        return obs, reward, done, info

    def time_interval(self):
        now = time.time()
        # self.logger.debug(self.event(event=LrEvents.SLEEP_START, step=self.current_step))

        delta_t = now - self.prev_sleep_time
        if delta_t - SEC_PER_STEP + SLEEP_CORRECTION < 0:
            time.sleep(SEC_PER_STEP - delta_t - SLEEP_CORRECTION)
            now = time.time()

        # self.logger.debug(self.event(event=LrEvents.SLEEP_END, step=self.current_step))
        self.prev_sleep_time = now

    def reset(self):
        # if self.log_this_episode:
        #     for handler in list(self.logger.handlers):
        #         handler.close()
        #         self.logger.removeHandler(handler)
        #     _fix_permissions(f"{self.log_dir}/env_events_log_{self.current_episode}.json")

        print(f"ENV: "
              f"E:{self.current_episode:4} "
              f"S:{self.current_step}  "
              f"total reward {self.total_reward:.2f}, "
              f"step time [ms] {mean(self.intervals_ms) if len(self.intervals_ms) != 0 else 0:.2f}, "
              f"{stdev(self.intervals_ms[1:]) if len(self.intervals_ms) != 0 else 0:.2f}, "
              f"{max(self.intervals_ms[1:]) if len(self.intervals_ms) != 0 else 0:.2f} : "
              f"{[i for i, e in enumerate(self.intervals_ms[1:]) if e >= max(self.intervals_ms[1:])]} : "
              f"{[i for i, e in enumerate(self.intervals_ms[1:]) if e > 10]}")

        # gym and RL
        self.current_step = 0
        self.current_episode += 1
        self.total_reward = 0

        # Operational
        self.prev_step_start = 0
        self.prev_sleep_time = 0
        self.intervals_ms = []
        self.episode_records = []

        self.episodes_from_last_log += 1
        self.log_this_episode = ((self.episodes_from_last_log % self.logs_episode_interval) == 0)
        # if self.log_this_episode:
        #     fh = logging.FileHandler(f"{self.log_dir}/env_events_log_{self.current_episode}.json")
        #     fh.setLevel(logging.DEBUG)
        #     fh.setFormatter(self.logger_format)
        #     self.logger.addHandler(fh)

        self.prev_sleep_time = time.time()

        return self.reset_agents()

    def render(self, mode='human', close=False):
        pass

    @abc.abstractmethod
    def close(self):
        """ TODO: add documentation """

    @abc.abstractmethod
    def get_tables(self):
        """ TODO: add documentation """

    @abc.abstractmethod
    def set_tables(self, new_tables):
        """ TODO: add documentation """

    @abc.abstractmethod
    def change_topo(self, new_topo):
        """ TODO: add documentation """

    @abc.abstractmethod
    def change_log_dir(self, new_log_dir):
        """ TODO: add documentation """

    @abc.abstractmethod
    def reset_episodes(self):
        """ TODO: add documentation """

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
