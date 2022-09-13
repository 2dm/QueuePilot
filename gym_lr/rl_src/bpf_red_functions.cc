#include <uapi/linux/if_ether.h>
#include <uapi/linux/in6.h>
#include <uapi/linux/ipv6.h>
#include <uapi/linux/bpf.h>
#include <bcc/proto.h>
#include <uapi/linux/tcp.h>
#include <uapi/linux/types.h>
#include <linux/sched.h>
#include <uapi/linux/ip.h>
#include <net/inet_sock.h>
#include <uapi/linux/ptrace.h>
#include <linux/ktime.h>
#include <net/pkt_cls.h>
#include <net/sch_generic.h>
#include <net/pkt_sched.h>
#include <net/red.h>



#define OBS_TX 0
#define OBS_Q_LEN 1
#define OBS_RX 2
#define OBS_DROPS 3
#define OBS_MARKED 4
#define OBS_CURRENT_Q 5
#define OBS_MAX_CURRENT_Q 6
#define OBS_MIN_CURRENT_Q 7
#define OBS_RED_QAVG 8
#define OBS_MARK_FILTER_RX_TCP 9
#define OBS_LENGTH 10


#define IP_TCP  6

BPF_ARRAY(obs, u64, OBS_LENGTH);
BPF_ARRAY(flows_cnt, uint32_t, 50);
BPF_ARRAY(ece_mode, long, 1);
BPF_HASH(my_enqueue_qdisc, u64, struct Qdisc *);
BPF_HASH(my_dequeue_qdisc, u64, struct Qdisc *);
BPF_HASH(red_q, u64, struct red_sched_data*);

// From /net/sched/sch_red.c
struct red_sched_data {
	u32			limit;		/* HARD maximal queue length */
	unsigned char		flags;
	struct timer_list	adapt_timer;
	struct Qdisc		*sch;
	struct red_parms	parms;
	struct red_vars		vars;
	struct red_stats	stats;
	struct Qdisc		*qdisc;
};


int rx_filter_no_acks(struct __sk_buff *skb)
{
    uint32_t my_dest_ip = DEST_IP;
    uint32_t obs_key = OBS_RX;

    u8 *cursor = 0;
    struct ethernet_t *ethernet = cursor_advance(cursor, sizeof(*ethernet));
    if (!(ethernet->type == 0x0800)) {
        return TC_ACT_OK;
    }

    struct ip_t *ip = cursor_advance(cursor, sizeof(*ip));
    if (ip->dst != my_dest_ip) {
            return TC_ACT_OK;
    }

    obs.atomic_increment(obs_key);

    return TC_ACT_OK;
}

int rx_filter_main(struct __sk_buff *skb)
{
    uint32_t my_dest_ip = DEST_IP;
    uint32_t obs_key = OBS_RX;

    u8 *cursor = 0;
    struct ethernet_t *ethernet = cursor_advance(cursor, sizeof(*ethernet));
    if (!(ethernet->type == 0x0800)) {
        return TC_ACT_OK;
    }

    struct ip_t *ip = cursor_advance(cursor, sizeof(*ip));
    if (ip->dst != my_dest_ip) {
            return TC_ACT_OK;
    }

    obs.atomic_increment(obs_key);

    return TC_ACT_OK;
}


static inline int tx_func(void* ctx, struct sk_buff* skb)
{
    uint32_t my_dest_ip = DEST_IP;
    uint32_t ifindex;

    u64* tx_count = 0;
    uint32_t obs_key = OBS_TX;


    struct net_device *dev;
    bpf_probe_read(&dev, sizeof(skb->dev), ((char*)skb) + offsetof(typeof(*skb), dev));
    bpf_probe_read(&ifindex, sizeof(dev->ifindex), &(dev->ifindex));

    if (ifindex != TX_INTERFACE) {
        return 0;
    }

    obs.atomic_increment(obs_key);

    return 0;
}

/* Attach to Kernel Tracepoints */
TRACEPOINT_PROBE(net, net_dev_start_xmit) {
    return tx_func(args, (struct sk_buff*)args->skbaddr);
}


static inline int update_stats(bool enqueue, int pkts)
{

    u64* q_len = 0;
    u64* drops = 0;
    u64* cur_q = 0;
    u64* max_cur_q = 0;
    u64* min_cur_q = 0;
    u64* marks = 0;
    uint32_t zero = 0;
    uint64_t obs_zero = 0;
    uint32_t obs_q_len_key = OBS_Q_LEN;
    uint32_t obs_drops_key = OBS_DROPS;
    uint32_t obs_curr_q__key = OBS_CURRENT_Q;
    uint32_t obs_max_curr_q__key = OBS_MAX_CURRENT_Q;
    uint32_t obs_min_curr_q__key = OBS_MIN_CURRENT_Q;
    uint32_t obs_marked_key = OBS_MARKED;
    u64 pid = bpf_get_current_pid_tgid();
    struct Qdisc **schp;

    if (enqueue) {
        schp = my_enqueue_qdisc.lookup(&pid);
    } else {
        schp = my_dequeue_qdisc.lookup(&pid);
    }

    if (schp == 0 || *schp == 0 ) {
        return 0;	// missed entry
    }
    struct Qdisc *sch = *schp;

    q_len = obs.lookup_or_try_init(&obs_q_len_key, &obs_zero);
    drops = obs.lookup_or_try_init(&obs_drops_key, &obs_zero);
    cur_q = obs.lookup_or_try_init(&obs_curr_q__key, &obs_zero);
    max_cur_q = obs.lookup_or_try_init(&obs_max_curr_q__key, &obs_zero);
    min_cur_q = obs.lookup_or_try_init(&obs_min_curr_q__key, &obs_zero);
    marks = obs.lookup_or_try_init(&obs_marked_key, &obs_zero);
    if ((q_len == 0) || (drops == 0) || (cur_q == 0) || (max_cur_q == 0) || (min_cur_q == 0) || (marks == 0)) {
        return 0;
    }

    u64 current_avgq = *q_len;
//    u64 current_q = sch->q.qlen;
    u64 current_q = sch->qstats.backlog;
    u64 max_q = *max_cur_q;
    u64 min_q = *min_cur_q;

    if (!enqueue) {
        for (int ii=pkts-1; ii>=0; --ii) {
            (*q_len) = (u64) (current_avgq +(current_q + ii - (current_avgq >> LOG_AVG_Q_WINDOW)));
        }
    }
    else {
        (*q_len) = (u64) (current_avgq +(current_q - (current_avgq >> LOG_AVG_Q_WINDOW)));
    }

//    (*drops) = sch->qstats.drops;
    struct red_sched_data* head = qdisc_priv(sch);
    (*drops) = head->stats.pdrop;

    (*marks) = head->stats.prob_mark +  head->stats.forced_mark;
    (*cur_q) = current_q;
    if (current_q > max_q) {
        (*max_cur_q) = current_q;
    } else if (current_q < min_q) {
        (*min_cur_q) = current_q;
    }

    return 0;
}


int kprobe__red_enqueue(struct pt_regs *ctx, struct sk_buff *skb, struct Qdisc *sch)
{
    u64 pid = bpf_get_current_pid_tgid();
    struct Qdisc *no_qdisc = NULL;

    if (sch->dev_queue->dev->ifindex != TX_INTERFACE) {
        my_enqueue_qdisc.update(&pid, &no_qdisc);
    } else {
        my_enqueue_qdisc.update(&pid, &sch);
    }
    struct red_sched_data* head = qdisc_priv(sch);

    red_q.lookup_or_try_init(&pid, &head);
    red_q.update(&pid, &head);

    return 0;
}

int kretprobe__red_enqueue(struct pt_regs *ctx)
{
    u64* max_cur_q = 0;
    uint64_t obs_zero = 0;
    int ret = PT_REGS_RC(ctx);

    u64 pid = bpf_get_current_pid_tgid();
    struct red_vars vars;
    struct red_parms params;
    struct red_sched_data** head_p;
    struct red_sched_data* head;
    head_p = red_q.lookup(&pid);
    if (head_p!=NULL){
        head = *head_p;
        if (head!=NULL) {
            bpf_probe_read(&vars, sizeof(vars), ((char*)head) + offsetof(typeof(*head), vars) );
            bpf_probe_read(&params, sizeof(params), ((char*)head) + offsetof(typeof(*head), parms) );
            uint32_t obs_key = OBS_RED_QAVG;
            u64* red_qavg_obs = obs.lookup_or_try_init(&obs_key, &obs_zero);
            if (red_qavg_obs == 0) {
                return 0;
            }
            (*red_qavg_obs) = vars.qavg>>params.Wlog;
        }
    }

    return update_stats(true, 0);
}


TRACEPOINT_PROBE(qdisc, qdisc_dequeue) {

    struct Qdisc *no_qdisc;
    int ifindex;
    ifindex = args->ifindex;
    no_qdisc = (struct Qdisc*) args->qdisc;
    int pkts = args->packets;

    u64 pid = bpf_get_current_pid_tgid();
    if ((ifindex == TX_INTERFACE) && (pkts > 0)) {
        my_dequeue_qdisc.update(&pid, &no_qdisc);
        return update_stats(false, pkts);
    }
    return 0;
}
