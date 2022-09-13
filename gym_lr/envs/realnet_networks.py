import abc
import subprocess
import time
import os
from gym_lr.rl_src.configurations import NUM_OF_CONNECTIONS, PACKET_SIZE_BYTES, UDP_BITRATE

# Return codes
from gym_lr.rl_src.system_configuration import EGRESS_IF, INGRESS_IF, DST_HOST_IP, KEY_FILE, DST_HOST_IF, SRC_HOST_IF, \
    SRC_HOST_IP, DST_HOST_USER, SRC_HOST_USER, INTERFACES, SENDER_IPERFS_SCRIPT_PATH, SERVER_IPERFS_SCRIPT_PATH, \
    SERVER_IPERFS_FCT_SCRIPT_PATH, NEXT_ROUTER_NAME, NEXT_ROUTER_IP, NEXT_ROUTER_CONGESTED_IF, TSHARK_LOG_DIR_PATH, \
    SENDER_IPERFS_CHANGING_LOAD_SCRIPT_PATH, SENDER_IPERFS_FCT_SCRIPT_PATH

PKILL_PROCESS_FOUND = 0
PKILL_PROCESS_NOT_FOUND = 1
success = 1
QDISC_SUCCESS = 0
QDISC_ERROR = 2  # fixme: generic error code ?? for qdisc not found/exists and cant nodify/etc. Parse and decide?


class RealNetBase:
    def __init__(self):
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} clsact")
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")
        for interface in INGRESS_IF:
            os.system(f'ethtool -K {interface} tso off gso off gro off')
        os.system(f'ethtool -K {EGRESS_IF} tso off gso off gro off')
        os.system('sysctl -w net.ipv4.ip_forward=1')
        os.system('sysctl -w net.ipv4.tcp_ecn=1')
        os.system('sysctl -w net.ipv4.tcp_ecn_fallback=0')

        self.popens = {}

    def setup_server_and_sender(self):
        ecn_cmds = [
            "sysctl -w net.ipv4.tcp_ecn=1",
            "sysctl -w net.ipv4.tcp_ecn_fallback=0"
        ]

        for ip_addr, net_if in zip([DST_HOST_IP, SRC_HOST_IP], [DST_HOST_IF, SRC_HOST_IF]):

            cmd = f'ssh -i {KEY_FILE} root@{ip_addr} ethtool -K {net_if} tso off gso off gro off'
            self.cmd_over_ssh(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            for ecn_cmd in ecn_cmds:
                cmd = f'ssh -i {KEY_FILE} root@{ip_addr} {ecn_cmd}'
                self.cmd_over_ssh(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def set_next_router_qdisc(self, max_buf=10000):
        cmds = [
            f'ssh -i {KEY_FILE} root@{NEXT_ROUTER_IP} ethtool -K {INTERFACES[NEXT_ROUTER_NAME]["ingress"]} tso off gso off gro off',
            f'ssh -i {KEY_FILE} root@{NEXT_ROUTER_IP} ethtool -K {INTERFACES[NEXT_ROUTER_NAME]["egress"]} tso off gso off gro off',
            f'ssh -i {KEY_FILE} root@{NEXT_ROUTER_IP} sysctl -w net.ipv4.tcp_ecn=1',
            f'ssh -i {KEY_FILE} root@{NEXT_ROUTER_IP} sysctl -w net.ipv4.tcp_ecn_fallback=0',
        ]

        for cmd in cmds:
            self.cmd_over_ssh(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.cmd_over_ssh(
            f'ssh -i {KEY_FILE} root@{NEXT_ROUTER_IP} tc qdisc replace dev {NEXT_ROUTER_CONGESTED_IF} root pfifo limit {max_buf}',
            allowed_exit_status=[QDISC_SUCCESS, QDISC_ERROR])

    def start_qdiscs(self, delay=10):
        self.cmd_over_ssh(
            f'ssh -i {KEY_FILE} root@{SRC_HOST_IP} tc qdisc del dev {SRC_HOST_IF} root ; '
            f'tc qdisc add dev {SRC_HOST_IF} root handle 8010: netem delay {delay}ms limit 10000',
            allowed_exit_status=[QDISC_SUCCESS])
        self.cmd_over_ssh(
            f'ssh -i {KEY_FILE} root@{DST_HOST_IP} tc qdisc del dev {DST_HOST_IF} root ; '
            f'tc qdisc add dev {DST_HOST_IF} root handle 8010: netem delay {delay}ms limit 10000',
            allowed_exit_status=[QDISC_SUCCESS])

    def start_server_iperfs(self, no_senders=NUM_OF_CONNECTIONS):
        iperf_cmd = f'ssh -i {KEY_FILE} {DST_HOST_USER}@{DST_HOST_IP} {SERVER_IPERFS_SCRIPT_PATH} {no_senders}'
        self.popens[f"server_iperfs_conn"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def start_server_fct_iperfs(self, no_senders=NUM_OF_CONNECTIONS):
        iperf_cmd = f'ssh -i {KEY_FILE} {DST_HOST_USER}@{DST_HOST_IP} {SERVER_IPERFS_FCT_SCRIPT_PATH} {no_senders}'
        self.popens[f"server_iperfs_conn"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def start_senders_iperfs(self, delay_start=0, no_senders=50, bg_traffic_mb=UDP_BITRATE, *args, **kwargs):
        iperf_cmd = f'ssh -i {KEY_FILE} {SRC_HOST_USER}@{SRC_HOST_IP} {SENDER_IPERFS_SCRIPT_PATH} {no_senders} {bg_traffic_mb} {DST_HOST_IP}'
        self.popens[f"sender_iperfs_conn"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def start_fct_senders_iperfs(self, no_senders=NUM_OF_CONNECTIONS, bg_traffic_mb=UDP_BITRATE, port_offset=0, tshark_log="", ep=0, msg_size="",  *args, **kwargs):
        iperf_cmd = f'ssh -i {KEY_FILE} root@{SRC_HOST_IP} mkdir -p {TSHARK_LOG_DIR_PATH}/{tshark_log}; ' \
                    f'chmod 777 {TSHARK_LOG_DIR_PATH}/{tshark_log}'
        self.popens[f"directory"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        capture_filter = "tcp portrange 5101-5200 and " \
                         "(tcp[tcpflags] & (tcp-syn) != 0 or tcp[tcpflags] & (tcp-fin) != 0 or tcp[tcpflags] & (tcp-rst) != 0)"
        iperf_cmd = f'ssh -i {KEY_FILE} root@{SRC_HOST_IP} tshark -s 100 -f \"{capture_filter}\" -i {SRC_HOST_IF} -w {TSHARK_LOG_DIR_PATH}/{tshark_log}/tshark_{msg_size}_{ep}.pcapng'
        self.popens[f"sender_tshark"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        iperf_cmd = f'ssh -i {KEY_FILE} {SRC_HOST_USER}@{SRC_HOST_IP} {SENDER_IPERFS_FCT_SCRIPT_PATH} {no_senders} {bg_traffic_mb} {DST_HOST_IP} {port_offset} {tshark_log} {ep} {msg_size}'
        self.popens[f"sender_iperfs_conn"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def start_changing_load_senders_iperfs(self,  *args, **kwargs):
        iperf_cmd = f'ssh -i {KEY_FILE} {SRC_HOST_USER}@{SRC_HOST_IP} {SENDER_IPERFS_CHANGING_LOAD_SCRIPT_PATH}'
        self.popens[f"sender_iperfs_conn"] = self.cmd_over_ssh(iperf_cmd, wait=0.3, timeout=-1,
                                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def kill_senders(self):
        for k in ["sender_iperfs_conn", "sender_tshark"]:
            try:
                p = self.popens[k]
                p.terminate()
            except KeyError:
                pass

        kill_cmd = f'ssh -i {KEY_FILE} {SRC_HOST_USER}@{SRC_HOST_IP} pkill iperf3'
        self.cmd_over_ssh(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          allowed_exit_status=[PKILL_PROCESS_FOUND,
                                               PKILL_PROCESS_NOT_FOUND])
        kill_cmd = f'ssh -i {KEY_FILE} {SRC_HOST_USER}@{SRC_HOST_IP} pkill nuttcp'
        self.cmd_over_ssh(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          allowed_exit_status=[PKILL_PROCESS_FOUND,
                                               PKILL_PROCESS_NOT_FOUND])

    def kill_tshark(self):
        kill_cmd = f'ssh -i {KEY_FILE} root@{SRC_HOST_IP} pkill tshark'
        self.cmd_over_ssh(kill_cmd, timeout=-1, allowed_exit_status=[PKILL_PROCESS_FOUND, PKILL_PROCESS_NOT_FOUND])

    def kill_servers(self):
        try:
            p = self.popens[f"server_iperfs_conn"]
            p.terminate()
        except KeyError:
            pass

        kill_cmd = f'ssh -i {KEY_FILE} {DST_HOST_USER}@{DST_HOST_IP} pkill iperf3'
        self.cmd_over_ssh(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          allowed_exit_status=[PKILL_PROCESS_FOUND,
                                               PKILL_PROCESS_NOT_FOUND])
        kill_cmd = f'ssh -i {KEY_FILE} {DST_HOST_USER}@{DST_HOST_IP} pkill nuttcp'
        self.cmd_over_ssh(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          allowed_exit_status=[PKILL_PROCESS_FOUND,
                                               PKILL_PROCESS_NOT_FOUND])

    def kill_connections(self):
        self.kill_senders()
        self.kill_servers()

        self.popens = {}

    def close(self):
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")

    @abc.abstractmethod
    def change_topo_params(self, new_params):
        """ Change topology parameters """

    def cmd_over_ssh(self, cmd, wait=0.0, timeout=1, stdin=None, stdout=None, stderr=None, allowed_exit_status=None):

        if allowed_exit_status is None:
            allowed_exit_status = [0]

        cmds_count = 0
        ret = -1

        while True:
            p = subprocess.Popen(cmd.split(), stdin=stdin, stdout=stdout, stderr=stderr)

            if wait > 0:
                time.sleep(wait)

            if timeout < 0:
                break

            cmds_count += 1
            out, err = None, None

            try:
                out, err = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"Timeout to ssh connection:{cmd}")
            else:
                ret = p.returncode
                if ret not in allowed_exit_status and cmds_count < 10:
                    print(f"Return code: {ret} cmd: {cmd}")
                    if out is not None or err is not None:
                        print(f"out: {out}; err:{err}")
                    time.sleep(2 ** cmds_count)
                else:
                    break

        return p


class RealPfifoNet(RealNetBase):
    def __init__(self, qdisc_params):
        super().__init__()
        self.setup_server_and_sender()
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")
        os.system(f"sudo tc qdisc add dev {EGRESS_IF} root handle 8010: pfifo limit {qdisc_params['buf_size']}")

    @abc.abstractmethod
    def change_topo_params(self, new_params):
        """ Change topology parameters """
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")
        os.system(f"sudo tc qdisc add dev {EGRESS_IF} root handle 8010: pfifo limit {new_params['buf_size']}")


class RealRedNet(RealNetBase):
    def __init__(self, qdisc_params):
        super().__init__()
        self.setup_server_and_sender()
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")
        ret = os.system(f"sudo tc qdisc replace dev {EGRESS_IF} root red limit {qdisc_params['limit']} "
                        f"min {qdisc_params['min']} max {qdisc_params['max']} avpkt {PACKET_SIZE_BYTES} "
                        f"burst {qdisc_params['burst']} "
                        f"bandwidth 1000mbit probability {qdisc_params['red_max_p']} ecn adaptive nodrop")
        if ret != 0:
            raise RuntimeError(f"tc qdisc RED error retrun = {ret}")

    def change_topo_params(self, new_params):
        """ Change topology parameters """
        os.system(f"sudo tc qdisc del dev {EGRESS_IF} root")
        os.system(f"sudo tc qdisc replace dev {EGRESS_IF} root red limit {new_params['limit']} "
                  f"min {new_params['min']} max {new_params['max']} avpkt {PACKET_SIZE_BYTES} "
                  f"burst {new_params['burst']} "
                  f"bandwidth 1000mbit probability {new_params['red_max_p']} ecn adaptive nodrop")


class RealRemotePfifoNet(RealNetBase):
    def __init__(self, qdisc_params):
        super().__init__()
        ret = os.system(f"sudo tc qdisc replace dev {EGRESS_IF} root pfifo limit {qdisc_params['buf_size']}")
        if ret != 0:
            raise RuntimeError(f"tc qdisc pfifo error retrun = {ret}")

    def change_topo_params(self, new_params):
        pass


class RealRemoteRedNet(RealNetBase):
    def __init__(self, qdisc_params):
        super().__init__()
        ret = os.system(f"sudo tc qdisc replace dev {EGRESS_IF} root red limit {qdisc_params['limit']} "
                        f"min {qdisc_params['min']} max {qdisc_params['max']} avpkt {PACKET_SIZE_BYTES} "
                        f"burst {qdisc_params['burst']} "
                        f"bandwidth 1000mbit probability 0.02 ecn")
        if ret != 0:
            raise RuntimeError(f"tc qdisc RED error retrun = {ret}")

    def change_topo_params(self, new_params):
        pass
