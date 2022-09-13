import inspect
import json
import os
import shutil
import stat
from enum import IntEnum
from enum import auto
from pathlib import Path

from gym_lr.rl_src.configurations import EPISODE_RUN_TIME_IN_SEC, STEPS_PER_EPISODE, STEPS_PER_TEST_EPISODE, \
    NUM_EPISODES_COLLECT_BEFORE_SGD, PACKET_SIZE_BYTES, SLEEP_CORRECTION, NUM_OF_EPISODES_TO_TRAIN, \
    NUM_OF_EPISODES_TO_TEST, LOG_AVG_Q_WINDOW, AVG_Q_WINDOW, TRAINING_DELAYS, TEST_DELAYS, TRAINING_NUM_OF_SENDERS, \
    TEST_NUM_OF_SENDERS, NUM_OF_CONNECTIONS, get_buffer_size, UDP_BITRATE
from gym_lr.rl_src.system_configuration import SRC_HOST_IP, SRC_HOST_IF, DST_HOST_IP, DST_HOST_IF, USERNAME


def write_params_log(log_dir="logs", **kwargs):
    log_rec = {
        "EPISODE_RUN_TIME_IN_SEC": EPISODE_RUN_TIME_IN_SEC,
        "STEPS_PER_EPISODE": STEPS_PER_EPISODE,
        "STEPS_PER_TEST_EPISODE": STEPS_PER_TEST_EPISODE,
        "NUM_EPISODES_COLLECT_BEFORE_SGD": NUM_EPISODES_COLLECT_BEFORE_SGD,
        "PACKET_SIZE_BYTES": PACKET_SIZE_BYTES,
        "SLEEP_CORRECTION": SLEEP_CORRECTION,
        "NUM_OF_EPISODES_TO_TRAIN": NUM_OF_EPISODES_TO_TRAIN,
        "NUM_OF_EPISODES_TO_TEST": NUM_OF_EPISODES_TO_TEST,
        "LOG_AVG_Q_WINDOW": LOG_AVG_Q_WINDOW,
        "AVG_Q_WINDOW": AVG_Q_WINDOW,

        "TRAINING_DELAYS": TRAINING_DELAYS,
        "TEST_DELAYS": TEST_DELAYS,
        "TRAINING_NUM_OF_SENDERS": TRAINING_NUM_OF_SENDERS,
        "TEST_NUM_OF_SENDERS": TEST_NUM_OF_SENDERS,
        "NUM_OF_CONNECTIONS": NUM_OF_CONNECTIONS,
        "UDP_BITRATE": UDP_BITRATE,

        "get_buffer_size": inspect.getsource(get_buffer_size).startswith('return'),
        "SRC_HOST_IP": SRC_HOST_IP,
        "SRC_HOST_IF": SRC_HOST_IF,
        "DST_HOST_IP": DST_HOST_IP,
        "DST_HOST_IF": DST_HOST_IF,
        "log_dir": log_dir,
    }
    for key, value in kwargs.items():
        log_rec[key] = value

    log_filename = f"{log_dir}/run_parameters.json"
    write_rec_to_file(log_rec, log_filename)


def log_ep_params(ep, bw, buffer_size, delay, senders):
    rec = {
        'Episode': ep,
        'Buffer size': buffer_size,
        'BW': bw,
        'Delay': delay,
        'Senders': senders,
    }
    return rec


def _fix_permissions(file):
    shutil.chown(file, USERNAME, USERNAME)
    os.chmod(file, os.stat(file).st_mode | stat.S_IWOTH)


def write_rec_to_file(record, file_name):
    with open(file_name, 'w') as fp:
        json.dump(record, fp, indent=4)
    _fix_permissions(file_name)


def is_port_in_use(port):
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        return s.connect_ex(('localhost', port)) == 0


def create_log_dir(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _fix_permissions(log_dir)


class LrEvents(IntEnum):
    STEP_START = auto()
    SET_MARK = auto()
    MARK_COMPLETED = auto()
    SLEEP_START = auto()
    SLEEP_END = auto()
    GET_OBS = auto()
    GOT_OBS = auto()
    STEP_COMPLETE = auto()

    def __str__(self):
        return self.name


class SwitchType(IntEnum):
    LEARNING = auto()
    RED = auto()
    CONST_THRESHOLD = auto()
    Droptail = auto()

    def __str__(self):
        return self.name


class StructuredMessage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return '%s' % json.dumps(self.kwargs)
