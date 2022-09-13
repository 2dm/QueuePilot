import concurrent
import itertools
import json
import glob
import os
import re
from builtins import range
from concurrent.futures import ALL_COMPLETED
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import cycle
from math import floor
from statistics import median
from turtledemo.chaos import plot

import psutil
from numpy import mean, std
from plotly.subplots import make_subplots
from statsmodels.distributions.empirical_distribution import ECDF

import numpy as np

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import utils
from gym_lr.rl_src.configurations import NUM_OF_CONNECTIONS, MBPS_TO_PKTS, GBPS_TO_PKTS_PER_STEP, PACKET_SIZE_BYTES, \
    SEC_PER_STEP, TEST_BUFFERS, TEST_DELAYS, TEST_NUM_OF_SENDERS, get_buffer_size, GBPS
from utils import LrEvents, _fix_permissions


def plot_episode_graph(log_dir="", episode=1, output_dir=None):
    fig = go.Figure()
    file = f"logs/{log_dir}/s1_episode_log_{episode}.json"
    file_ep_params = f"logs/{log_dir}/s1_ep_params_{episode}.json"

    data = pd.read_json(file)
    params_data = pd.read_json(file_ep_params, typ='series')

    bw = params_data['BW']
    buf_size = params_data['Buffer size']
    senders = params_data['Senders']
    delay = params_data['Delay']

    reward = sum(data['Step reward'])

    fig.add_trace(go.Scatter(y=data['Action'] / 100, mode='lines', name="Action"))
    fig.add_trace(
        go.Scatter(y=data['TX'] / floor(bw * SEC_PER_STEP), mode='lines', name="Sent packets"))
    fig.add_trace(
        go.Scatter(y=data['Q length'] / buf_size, mode='lines', name="Queue length"))
    fig.add_trace(go.Scatter(y=data['Dropped'], mode='lines', name="Dropped"))
    fig.add_trace(go.Scatter(y=data['RX'] / floor(bw * SEC_PER_STEP), mode='lines',
                             name="Received packets"))
    fig.add_trace(go.Scatter(y=data['Step reward'], mode='lines', name="Step Reward"))
    fig.add_trace(go.Scatter(y=(data['Marked'] / data['RX']).fillna(0), mode='lines', name="Step marking"))
    fig.add_trace(go.Scatter(y=(data['RED qavg'] / buf_size).fillna(0), mode='lines', name="RED avg. queue"))
    fig.add_trace(go.Scatter(y=data['CurrentQ length'] / buf_size, mode='lines', name="Current Q length", ))
    fig.add_trace(go.Scatter(y=data['Max CurrentQ length'] / buf_size, mode='lines', name="Max Q length", ))
    fig.add_trace(go.Scatter(y=data['Min CurrentQ length'] / buf_size, mode='lines', name="Min Q length"))


    # === Single switch detailed title ===
    fig.update_layout(xaxis_title=f"Step ({int(SEC_PER_STEP * 1000)}ms)",
                      title=f"{log_dir} Episode {episode}:\t"
                            f"Buffer size: {int(buf_size)}pkts\t"
                            f"Senders: {senders}\t"
                            f"RTT: {2 * delay}ms\t"
                            f"BW: {int(bw)}pps\t"
                            f"Reward: {reward:.2f}")

    fig.update_yaxes(range=[-0.1, 1.2])
    file_name = f"logs/graphs/{log_dir}_episode_{episode}.html" if output_dir is None else f"{output_dir}/episode_{episode}.html"
    fig.write_html(file_name)
    _fix_permissions(file_name)


if __name__ == '__main__':
    plot_episode_graph(log_dir="", episode=2)
