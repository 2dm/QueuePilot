from gym_lr.switch.switch_base import SwitchBase
from utils import LrEvents, SwitchType


class TailDropSwitch(SwitchBase):

    def __init__(self, config):
        config['switch_type'] = SwitchType.Droptail
        super().__init__(config)

    def apply_action(self, action=None):
        self.current_step += 1
        # self.logger.debug(self.event(event=LrEvents.SET_MARK, step=self.current_step))
        # self.logger.debug(self.event(event=LrEvents.MARK_COMPLETED, step=self.current_step))
