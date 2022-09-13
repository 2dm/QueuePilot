import os

USERNAME = ""

# Hosts
HOST1_USER = ""
HOST1_IP = "192.168.1.1"
HOST1_IF = ""

HOST2_USER = ""
HOST2_IP = "192.168.4.1"
HOST2_IF = ""
# add more hosts if you need

# Routers
R1_INGRESS_IF = [""]        # This router
R1_EGRESS_IF = ""           # egress to R2
R2_INGRESS_IF = []
R2_EGRESS_IF = ""

INTERFACES = {
    "router_hostname": {
        "ingress": R1_INGRESS_IF,
        "egress": R1_EGRESS_IF
    },
    "other_router_hostname": {
        "ingress": R2_INGRESS_IF,
        "egress": R2_EGRESS_IF
    }
}
# Path to ssh key file to other hosts
KEY_FILE = ""

# =========================================================================
# TCP Source
SRC_HOST_USER = HOST1_USER
SRC_HOST_IP = HOST1_IP
SRC_HOST_IF = HOST1_IF
# Paths to scripts in source host. Edit this
# For exmaple:
SENDER_IPERFS_SCRIPT_PATH = f"/home/{SRC_HOST_USER}/Desktop/start_iperf_senders.sh"
SENDER_IPERFS_FCT_SCRIPT_PATH = f"/home/{SRC_HOST_USER}/Desktop/start_iperf_senders_fct.sh"
SENDER_IPERFS_CHANGING_LOAD_SCRIPT_PATH = f"/home/{SRC_HOST_USER}/Desktop/start_iperf_senders_dynamic.sh"
TSHARK_LOG_DIR_PATH = f"/home/{SRC_HOST_USER}/Desktop/"

# TCP Destination
DST_HOST_USER = HOST2_USER
DST_HOST_IP = HOST2_IP
DST_HOST_IF = HOST2_IF
# Paths to scripts in destination host. Edit this
# For exmaple:
SERVER_IPERFS_SCRIPT_PATH = f"/home/{DST_HOST_USER}/Desktop/start_iperf_servers.sh"
SERVER_IPERFS_FCT_SCRIPT_PATH = f"/home/{DST_HOST_USER}/Desktop/start_iperf_servers_fct.sh"

# Next router (R_2)
NEXT_ROUTER_NAME = "R2"
NEXT_ROUTER_IP = ""
NEXT_ROUTER_CONGESTED_IF = R2_EGRESS_IF


# This router
INGRESS_IF = INTERFACES[os.uname()[1]]["ingress"]
EGRESS_IF = INTERFACES[os.uname()[1]]["egress"]

