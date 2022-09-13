import ctypes as ct
from math import floor

from gym_lr.rl_src.configurations import AVG_Q_WINDOW
from gym_lr.switch.switch_base import SwitchBase, OBS_Q_LEN
from utils import LrEvents


class ConstSwitch(SwitchBase):

    def __init__(self, config):
        super().__init__(config)

        self.const_threshold = floor(config["const_threshold_ratio"] * self.buf_size)

    def apply_action(self, action):
        self.current_step += 1
        # self.logger.debug(self.event(event=LrEvents.SET_MARK, step=self.current_step))

        apply_action_prob = action if ((self.prev_sampled_obs[OBS_Q_LEN]) / AVG_Q_WINDOW) >= self.const_threshold else 0
        self.ece_mode_table[0] = ct.c_int(apply_action_prob)
        self.action = int(apply_action_prob)

        # self.logger.debug(self.event(event=LrEvents.MARK_COMPLETED, step=self.current_step))
