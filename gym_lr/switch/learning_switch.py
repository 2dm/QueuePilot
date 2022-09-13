import ctypes as ct
from gym_lr.switch.switch_base import SwitchBase
from utils import LrEvents, SwitchType


class LearningSwitch(SwitchBase):

    def __init__(self, config):
        config['switch_type'] = SwitchType.LEARNING
        super(LearningSwitch, self).__init__(config)

    def apply_action(self, action):
        self.current_step += 1
        # self.logger.debug(self.event(event=LrEvents.SET_MARK, step=self.current_step))

        action_map = [0, 1, 5, 10, 50, 100]  # Mapped to [0, 0.01, 0.05, ..., 1]
        p = action_map[action]
        self.ece_mode_table[0] = ct.c_uint32(p)
        self.action = p

        # self.logger.debug(self.event(event=LrEvents.MARK_COMPLETED, step=self.current_step))

