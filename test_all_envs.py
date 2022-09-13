import datetime
import itertools
import os
import random
import sys
import time
from math import ceil
import ray
import test
import test_ared
import test_droptail
from gym_lr.rl_src.configurations import SINGLE_SWITCH_EVAL_TOPO_PARAMS, SINGLE_RED_SWITCH_EVAL_TOPO_PARAMS, \
    PACKET_SIZE_BYTES, get_buffer_size, TEST_DELAYS, TEST_NUM_OF_SENDERS, TEST_BUFFERS, LSTM_SIZE


def run_test(test_label, checkpoint):
    real_net, agent, env = test.build_setup(label=test_label, topo_params=SINGLE_SWITCH_EVAL_TOPO_PARAMS,
                                            agent_checkpoint=checkpoint)

    delays = TEST_DELAYS
    senders = TEST_NUM_OF_SENDERS
    buffers = [TEST_BUFFERS]
    bw = 1000

    for (delay, no_senders, buf) in list(itertools.product(delays, senders, buffers)):
        buf = get_buffer_size(bw=bw, delay=delay, senders=no_senders) if buf is None else buf
        eval_topo_parmas = {
            "delay": delay,
            "buf_size": buf,
            "bw": bw,
            "no_senders": no_senders,
        }

        label = f"{test_label}/rl_bw_{bw}_delay_{delay}_buf_{buf}_senders_{no_senders}"
        test.change_setup_params(label, real_net, agent, env, eval_topo_parmas)
        print(label)

        seed = int(random.uniform(0, 2 ** 32 - 1))
        agent.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.seed(seed)))
        random.seed(seed)

        test.test_single_env(eval_topo_parmas, real_net, agent, env, real_net.start_senders_iperfs)
        env.reset_episodes()
        # agent.workers.foreach_worker(lambda ev: ev.foreach_env(lambda env: env.reset_episodes()))

    test.close_all(env, real_net, agent)

    # ARED

    real_net, agent, env = test_ared.build_setup(label=test_label, topo_params=SINGLE_RED_SWITCH_EVAL_TOPO_PARAMS)

    for min_thresh_r in [0.1, 0.2, 0.3]:
        for (delay, no_senders, buf) in list(itertools.product(delays, senders, buffers)):
            buf = get_buffer_size(bw=bw, delay=delay, senders=no_senders) if buf is None else buf
            eval_topo_parmas = {
                "delay": delay,
                "buf_size": buf,
                "bw": bw,
                "no_senders": no_senders,
                "limit": buf * PACKET_SIZE_BYTES,
                "min": buf * PACKET_SIZE_BYTES * min_thresh_r,
                "max": buf * PACKET_SIZE_BYTES * min_thresh_r * 3,
                "burst": 1 + ceil(buf * min_thresh_r),
                "red_max_p": 0.1,
            }

            label = f"{test_label}/ared_min_{int(min_thresh_r * 100)}_bw_{bw}_delay_{delay}_buf_{buf}_senders_{no_senders}"
            test_ared.change_setup_params(label, real_net, agent, env, eval_topo_parmas)
            print(label)

            seed = int(random.uniform(0, 2 ** 32 - 1))
            agent.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.seed(seed)))
            random.seed(seed)

            test_ared.test_single_env(eval_topo_parmas, real_net, agent, env, real_net.start_senders_iperfs)
            env.reset_episodes()
            # agent.workers.foreach_worker(lambda ev: ev.foreach_env(lambda env: env.reset_episodes()))

    test_ared.close_all(env, real_net, agent)

    # Droptail

    real_net, agent, env = test_droptail.build_setup(label=test_label, topo_params=SINGLE_SWITCH_EVAL_TOPO_PARAMS)

    for (delay, no_senders, buf) in list(itertools.product(delays, senders, buffers)):
        buf = get_buffer_size(bw=bw, delay=delay, senders=no_senders) if buf is None else buf
        eval_topo_parmas = {
            "delay": delay,
            "buf_size": buf,
            "bw": bw,
            "no_senders": no_senders,
        }

        label = f"{test_label}/droptail_bw_{bw}_delay_{delay}_buf_{buf}_senders_{no_senders}"
        test.change_setup_params(label, real_net, agent, env, eval_topo_parmas)
        print(label)

        seed = int(random.uniform(0, 2 ** 32 - 1))
        agent.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.seed(seed)))
        random.seed(seed)

        test_droptail.test_single_env(eval_topo_parmas, real_net, agent, env, real_net.start_senders_iperfs)
        env.reset_episodes()
        # agent.workers.foreach_worker(lambda ev: ev.foreach_env(lambda env: env.reset_episodes()))

    test.close_all(env, real_net, agent)


if __name__ == '__main__':
    os.nice(-20)
    # label = "test_"
    label = sys.argv[1]
    cp = sys.argv[2]

    start = time.time()
    print(f'Starting multi-topology test real run at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")}')
    ray.init()
    run_test(label, checkpoint=cp)
    ray.shutdown()

    print(f'Completed at {datetime.datetime.now().isoformat(sep=" ", timespec="seconds")} ; '
          f'{str(datetime.timedelta(seconds=time.time() - start))}')
