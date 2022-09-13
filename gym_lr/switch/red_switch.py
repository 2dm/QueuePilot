from gym_lr.switch.switch_base import SwitchBase
from utils import LrEvents, SwitchType


class REDSwitch(SwitchBase):

    def __init__(self, config):
        config['switch_type'] = SwitchType.RED
        super(REDSwitch, self).__init__(config)

        self.red_min_threshold = config["red_min_threshold"]
        self.red_max_threshold = config["red_max_threshold"]
        self.limit = config["limit"]
        self.max_p = config["red_max_p"]
        self.burst = config["burst"]
        self.red_marked_count = 0
        self.buf_size = self.limit

    def apply_action(self, action=None):
        self.current_step += 1
        # self.logger.debug(self.event(event=LrEvents.SET_MARK, step=self.current_step))
        # self.logger.debug(self.event(event=LrEvents.MARK_COMPLETED, step=self.current_step))

    def log_interval(self, *args):
        super(REDSwitch, self).log_interval(*args)
        log_rec = {
            'red_min_threshold': self.red_min_threshold,
            'red_max_threshold': self.red_max_threshold,
            'red_max_p': self.max_p,
            'limit': self.limit
        }
        self.episode_records[-1].update(log_rec)

    def change_topo(self, new_topo):
        super(REDSwitch, self).change_topo(new_topo)
        self.limit = new_topo["limit"]
        self.buf_size = new_topo["limit"]
        self.max_p = new_topo["red_max_p"]
        self.burst = new_topo["burst"]
        self.red_min_threshold = new_topo["min"]
        self.red_max_threshold = new_topo["max"]
