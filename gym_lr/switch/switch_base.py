import abc
import ipaddress
import json
# import logging
import errno
import warnings
from ctypes import c_int, c_ulong, c_uint64, c_uint32
from math import ceil, sqrt, floor

from gym import spaces
import numpy as np

from gym_lr.rl_src.configurations import MBPS_TO_PKTS, SEC_PER_STEP, NUM_OF_CONNECTIONS, AVG_Q_WINDOW, LOG_AVG_Q_WINDOW, \
    PACKET_SIZE_BYTES
from gym_lr.envs.realnet_networks import DST_HOST_IP
from utils import StructuredMessage, LrEvents, _fix_permissions, SwitchType, write_rec_to_file, log_ep_params
from bcc import BPF

from pyroute2 import IPRoute, NetlinkError
from inspect import currentframe, getframeinfo

OBS_TX = 0
OBS_Q_LEN = 1
OBS_RX = 2
OBS_DROPS = 3
OBS_MARKED = 4
OBS_CURRENT_Q = 5
OBS_MAX_CURRENT_Q = 6
OBS_MIN_CURRENT_Q = 7
OBS_RED_QAVG = 8
OBS_MARK_FILTER_RX_TCP = 9
OBS_LENGTH = 10


class SwitchBase:

    def __init__(self, config):
        super(SwitchBase, self).__init__()

        self.switch_name = config["switch_name"]
        self.log_dir = config["log_dir"]
        self.bw = floor(config["bw"] * MBPS_TO_PKTS)
        self.buf_size = config["buf_size"]
        self.delay = config["delay"]
        self.logs_episode_interval = config["logs_episode_interval"]
        self.create_bpf = config["create_bpf"]
        self.egress_if = config["egress_if"]
        self.ingress_if = config["ingress_if"]
        self.switch_type = config["switch_type"]
        self.num_of_steps = config["num_of_steps"]
        self.senders = config["no_senders"]

        obs_low_bound = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        obs_high_bound = np.array([1, 1, 30, 3, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low_bound, obs_high_bound, dtype=np.float32)
        self.action_space = spaces.Discrete(n=6)

        self.prev_sampled_obs = [0] * OBS_LENGTH
        self.get_flows = False
        self.prev_flows = np.array([0] * NUM_OF_CONNECTIONS)

        self.episode_dropped_pkts = 0
        self.episode_tx_pkts = 0
        self.episode_rx_pkts = 0

        self.action = 0
        self.current_step = 0
        self.current_episode = 0

        self.total_reward = 0

        # self.logger = logging.getLogger(f'event_log')
        # self.logger.setLevel(logging.DEBUG)
        self.event = StructuredMessage

        self.log_this_episode = False
        self.episode_records = []
        self.events_records = []
        self.log_filename = f"{self.log_dir}/{self.switch_name}_episode_log_{{}}.json"
        self.events_filename = f"{self.log_dir}/{self.switch_name}_event_log_{{}}.json"
        self.ep_params_filename = f"{self.log_dir}/{self.switch_name}_ep_params_{{}}.json"

        self.q_error = 0

        if self.create_bpf:
            try:
                self.ipr = IPRoute()
                print(f"looking for egress IF {self.egress_if}")
                self.mark_if_index = self.ipr.link_lookup(ifname=f"{self.egress_if}")[0]
                self.rx_if_index = []
                for rx_interface in iter(self.ingress_if):
                    print(f"looking for ingress IF {rx_interface}")
                    self.rx_if_index.append(self.ipr.link_lookup(ifname=f"{rx_interface}")[0])
                print(f"TX IF {self.egress_if} index is: {self.mark_if_index}")
                print(f"RX IF {self.ingress_if} index is: {self.rx_if_index}")

                if self.switch_type == SwitchType.LEARNING or self.switch_type == SwitchType.Droptail:
                    bpf_file = 'gym_lr/rl_src/bpf_functions.cc'
                else:
                    bpf_file = 'gym_lr/rl_src/bpf_red_functions.cc'

                with open(bpf_file, 'r') as f:
                    bpf_text = f.read()

                bpf_text = bpf_text.replace('DEST_IP', str(int(ipaddress.ip_address(DST_HOST_IP))))
                bpf_text = bpf_text.replace('TX_INTERFACE', str(self.mark_if_index))
                bpf_text = bpf_text.replace('LOG_AVG_Q_WINDOW', str(LOG_AVG_Q_WINDOW))

                self.bpf = BPF(text=bpf_text)
                self.obs_table = self.bpf['obs']
                self.ece_mode_table = self.bpf['ece_mode']
                self.rx_flow_cnt = self.bpf['flows_cnt']

                # RX filters
                for rx_if_idx, func_name in zip(self.rx_if_index, ["rx_filter_main", "rx_filter_no_acks"]):
                    try:
                        self.ipr.tc("del", "ingress", rx_if_idx)
                    except NetlinkError as e:
                        if e.args[0] == 2:  # No such file or directory
                            pass

                    mark_fn = self.bpf.load_func(func_name, BPF.SCHED_CLS)
                    self.ipr.tc("add", "ingress", rx_if_idx, "ffff:")
                    self.ipr.tc("add-filter", "bpf", rx_if_idx, ":1", fd=mark_fn.fd, name=mark_fn.name,
                                parent="ffff:", classid=1, action="ok")

                # TX marking filter
                try:
                    self.ipr.tc("del", "clsact", self.mark_if_index)
                except NetlinkError as e:
                    if e.args[0] == 2:  # No such file or directory
                        pass
                if self.switch_type == SwitchType.LEARNING:
                    mark_fn = self.bpf.load_func("ce_mark_func", BPF.SCHED_CLS)
                    self.ipr.tc("add", "clsact", self.mark_if_index)
                    self.ipr.tc("add-filter", "bpf", self.mark_if_index, ":1", fd=mark_fn.fd, name=mark_fn.name,
                                parent="ffff:fff3", classid=1, direct_action=True)

                self.step_keys = [OBS_MAX_CURRENT_Q, OBS_MIN_CURRENT_Q]
                self.step_values = [c_uint64(0), c_uint64(9999999999)]
                ct_int_array = c_int * len(self.step_keys)
                ct_u64_array = c_uint64 * len(self.step_keys)
                self.step_ct_keys = ct_int_array(*self.step_keys)
                self.step_ct_values = ct_u64_array(*self.step_values)

            except KeyboardInterrupt:
                print("bpf error")
                return

    @abc.abstractmethod
    def apply_action(self, action):
        """ step function"""

    def reset_obs(self):
        sampled_obs = {}
        for k, v in self.obs_table.items_lookup_batch():
            sampled_obs[k] = v
        self.prev_sampled_obs = sampled_obs.copy()
        if self.create_bpf:
            self.rx_flow_cnt.clear()
            self.obs_table.items_update_batch(ct_keys=self.step_ct_keys, ct_values=self.step_ct_values)
            # for key in self.rx_flow_cnt:
            #     self.rx_flow_cnt[key] = c_int(0)

    def finalize_step(self):
        # Sample observations
        # self.logger.debug(self.event(event=LrEvents.GET_OBS, step=self.current_step))

        # start = time.time()
        sampled_obs = {}
        ite = self.obs_table.items_lookup_batch()
        for k, v in ite:
            sampled_obs[k] = v

        self.obs_table.items_update_batch(ct_keys=self.step_ct_keys, ct_values=self.step_ct_values)

        diff_flows = None
        if self.get_flows:
            sampled_flows = np.fromiter(map(lambda x: x.value, self.rx_flow_cnt.values()), dtype=int)
            diff_flows = np.subtract(sampled_flows, self.prev_flows)
            self.prev_flows = sampled_flows.copy()

        # self.logger.debug(self.event(event=LrEvents.GOT_OBS, step=self.current_step))

        tx_pkts = sampled_obs[OBS_TX] - self.prev_sampled_obs[OBS_TX]
        rx_pkts = sampled_obs[OBS_RX] - self.prev_sampled_obs[OBS_RX]
        dropped_pkts = sampled_obs[OBS_DROPS] - self.prev_sampled_obs[OBS_DROPS]
        marked_pkts = sampled_obs[OBS_MARKED] - self.prev_sampled_obs[OBS_MARKED]
        mark_f_rx_tcp = sampled_obs[OBS_MARK_FILTER_RX_TCP] - self.prev_sampled_obs[OBS_MARK_FILTER_RX_TCP]
        q_len = sampled_obs[OBS_Q_LEN] / AVG_Q_WINDOW

        if self.switch_type == SwitchType.RED:
            q_len_pkts = q_len/PACKET_SIZE_BYTES
        else:
            q_len_pkts = q_len

        self.episode_tx_pkts += tx_pkts
        self.episode_dropped_pkts += dropped_pkts
        self.episode_rx_pkts += rx_pkts

        obs = [min(tx_pkts / floor(self.bw * SEC_PER_STEP), float(self.observation_space.high[0])),
               min(q_len / self.buf_size, float(self.observation_space.high[1])),
               min(q_len_pkts / floor(self.bw * SEC_PER_STEP), float(self.observation_space.high[2])),
               min(rx_pkts / floor(self.bw * SEC_PER_STEP), float(self.observation_space.high[3])),
               min(dropped_pkts / floor(self.bw * SEC_PER_STEP), float(self.observation_space.high[4])),
               min(marked_pkts / rx_pkts if rx_pkts > 0 else 1 if marked_pkts > 0 else 0, float(self.observation_space.high[5]))]

        reward = (obs[0] ** 2) / ((1 + obs[2])**(1./2)) if dropped_pkts == 0 else -1
        self.total_reward += reward

        # Check for run issues
        if ((q_len / self.buf_size) > 1) and (q_len > self.q_error):
            warnings.warn(f"Buffer size error: {self.switch_name} step: {self.current_step} "
                          f"q_len={q_len:.2f}, "
                          f"buf={self.buf_size}, "
                          f"previous q_len = {self.prev_sampled_obs[OBS_Q_LEN]:.2f}", RuntimeWarning)
            self.q_error = q_len
        if (dropped_pkts > 0 and rx_pkts == 0) or (rx_pkts > 0 and dropped_pkts / rx_pkts > 1):
            warnings.warn(f"Drop/RX sampling issue: {self.switch_name} step: {self.current_step} "
                          f"dropped={dropped_pkts}, "
                          f"received={rx_pkts}, "
                          f"sampled dropped={sampled_obs[OBS_DROPS]}, "
                          f"previous dropped={self.prev_sampled_obs[OBS_DROPS]}", RuntimeWarning)

        self.prev_sampled_obs = sampled_obs.copy()

        done = self.current_step >= self.num_of_steps

        if self.log_this_episode:
            self.log_interval(self.action, tx_pkts, rx_pkts, dropped_pkts, q_len, marked_pkts, mark_f_rx_tcp, reward,
                              sampled_obs, obs)
            if done:
                log_filename = self.log_filename.format(self.current_episode)
                events_filename = self.events_filename.format(self.current_episode)
                ep_params_filename = self.ep_params_filename.format(self.current_episode)

                write_rec_to_file(record=self.episode_records, file_name=log_filename)
                self.episode_records = []
                write_rec_to_file(record=self.events_records, file_name=events_filename)
                self.events_records = []
                write_rec_to_file(
                    record=log_ep_params(self.current_episode, self.bw, self.buf_size, self.delay, self.senders),
                    file_name=ep_params_filename)

                # for i in [1, 5, 10, 50]:
                #     print(f"dist{i}:")
                #     print(self.bpf[f'dist{i}'].print_linear_hist("interval"))

        return np.array(obs).astype(np.float32), reward, done, {}

    def log_interval(self, action, tx_pkts, rx_pkts, dropped_pkts, q_len, marked_pkts, mark_f_rx_tcp,
                     reward, sampled_obs, final_obs, flows=None):
        log_rec = {
            'Step': self.current_step,
            'Action': action,
            'TX': tx_pkts,
            'RX': rx_pkts,
            'Dropped': dropped_pkts,
            'Q length': q_len,
            'CurrentQ length': sampled_obs[OBS_CURRENT_Q],
            'Max CurrentQ length': sampled_obs[OBS_MAX_CURRENT_Q],
            'Min CurrentQ length': sampled_obs[OBS_MIN_CURRENT_Q],
            'Step reward': reward,
            'Marked': marked_pkts,
            'RX TCP on filter': mark_f_rx_tcp,
            'Flows': flows,
            'RED qavg': sampled_obs[OBS_RED_QAVG],
        }
        self.episode_records.append(log_rec)

        event_rec = {
            'Step': self.current_step,
            'Obs': final_obs,
            'Action': action,
            'Step reward': reward,
            'Sampled TX': sampled_obs[OBS_TX],
            'TX': tx_pkts,
            'Sampled RX': sampled_obs[OBS_RX],
            'RX': rx_pkts,
            'Q length': q_len,
            'CurrentQ length': sampled_obs[OBS_CURRENT_Q],
            'Max CurrentQ length': sampled_obs[OBS_MAX_CURRENT_Q],
            'Min CurrentQ length': sampled_obs[OBS_MIN_CURRENT_Q],
            'Sampled dropped': sampled_obs[OBS_DROPS],
            'Dropped': dropped_pkts,
            'Marked': marked_pkts,
            'RX TCP on filter': mark_f_rx_tcp,
        }
        self.events_records.append(event_rec)

    def reset(self):
        print(f"{self.switch_name}   "
              f"E:{self.current_episode:4} "
              f"S:{self.current_step}  "
              f"Sent {self.episode_tx_pkts}, "
              f"Dropped {self.episode_dropped_pkts}, "
              f"Received {self.episode_rx_pkts}, "
              f"last q {self.prev_sampled_obs[OBS_Q_LEN]:.2f}, "
              f"Total reward {self.total_reward:.2f}")

        self.episode_dropped_pkts = 0
        self.episode_tx_pkts = 0
        self.episode_rx_pkts = 0

        self.prev_flows = np.array([0] * NUM_OF_CONNECTIONS)

        self.episode_dropped_pkts = 0
        self.episode_tx_pkts = 0
        self.episode_rx_pkts = 0

        self.prev_sampled_obs = [0] * OBS_LENGTH
        self.action = 0
        self.current_step = 0
        self.current_episode += 1

        self.total_reward = 0
        self.q_error = 0

        self.log_this_episode = ((self.current_episode % self.logs_episode_interval) == 0)
        self.episode_records = []
        self.events_records = []

        initial_obs = [0] * self.observation_space.shape[0]

        return np.array(initial_obs).astype(np.float32)

    def change_log_dir(self, new_log_dir):
        self.log_dir = new_log_dir
        self.log_filename = f"{self.log_dir}/{self.switch_name}_episode_log_{{}}.json"
        self.events_filename = f"{self.log_dir}/{self.switch_name}_event_log_{{}}.json"
        self.ep_params_filename = f"{self.log_dir}/{self.switch_name}_ep_params_{{}}.json"

    def change_topo(self, new_topo):
        self.bw = floor(new_topo["bw"] * MBPS_TO_PKTS)
        self.buf_size = new_topo["buf_size"]
        self.delay = new_topo["delay"]
        self.senders = new_topo["no_senders"]
        print(f"{self.switch_name}: New topo: delay {self.delay}, buf {self.buf_size}, "
              f"bw {self.bw}, senders {self.senders}")
        try:
            self.obs_table.clear()
            q_len = self.obs_table[OBS_Q_LEN].value
            if q_len > 0:
                warnings.warn(f"Change topo qlen {q_len}", RuntimeWarning)
            return q_len
        except AttributeError:
            pass

        return 0

    def close(self):
        if self.create_bpf:
            try:
                self.ipr.tc("del", "clsact", self.mark_if_index)
            except (AttributeError, NetlinkError) as e:
                if e.args[0] == errno.EINVAL:
                    pass
                else:
                    print(e.args)

            for rx_if in self.rx_if_index:
                try:
                    self.ipr.tc("del", "ingress", rx_if)
                except IndexError:
                    pass
                except (AttributeError, NetlinkError) as e:
                    if e.args[0] == errno.EINVAL:
                        pass
                    else:
                        frameinfo = getframeinfo(currentframe())
                        print(frameinfo.filename, frameinfo.function, frameinfo.lineno, end='')
                        print(e.args)

            try:
                self.bpf.cleanup()
            except Exception as e:
                frameinfo = getframeinfo(currentframe())
                print(frameinfo.filename, frameinfo.function, frameinfo.lineno, end='')
                print(e)

    def get_tables(self):
        print(f"Getting tables {self.switch_name}: "
              f"{self.obs_table} "
              f"{self.ece_mode_table} ")
        return {"obs_table": self.obs_table,
                "ece_mode_table": self.ece_mode_table}

    def set_tables(self, new_tables):
        self.obs_table = new_tables["obs_table"]
        self.ece_mode_table = new_tables["ece_mode_table"]
        print(
            f"Setting tables {self.switch_name}: "
            f"{self.obs_table} "
            f"{self.ece_mode_table} ")

    def reset_episodes(self):
        self.current_episode = 0
