# QueuePilot 

A reinforcement learning-based active queue management (AQM) that enables small buffers in backbone routers.

## Setup
QueuePilot was tested on Ubuntu 20.04 and python 3.9. 
A basic configuration includes a source host and a destination host, connected through two routing PCs. 

The file `environment.yml` contains the python dependencies.

```
sudo apt install -y ethtool openssh-server openssh-client
conda env create -f environment.yml
```
QueuePilot uses [BCC](https://github.com/iovisor/bcc) to create and load eBPF code and get access to kernel space.
Use [BCC's installation instructions](https://github.com/iovisor/bcc/blob/master/INSTALL.md#ubuntu---source) to build BCC. 

**Source and destination hosts**
1. Install dependencies:
`sudo apt install -y ethtool openssh-server nuttcp`
2. Build iperf3 from [iperf3 repository](https://github.com/esnet/iperf)
3. Copy the scripts from `scripts/src` and `scripts/dst` respectively.
4. Create ssh key-file to connect from the router to the source and destination hosts. 
Changing ECN properties and creating qdisc requires elevated permissions. 



Update all topology parameters in `gym_lr/rl_src/system_configuration.py`

## Usage
eBPF requires elevated permissions, so the following should run with sudo or as root. 
This is in addition to the source/destination hosts commands, so consider your system's security.

`conda activate queue_pilot`

**Train:**
`python train.py [train label]`

**Test:**
`python test_all_envs.py <test label> <checkpoint_path>`
Addtional tests can be found in `test.py`

**Output files** are available in `QueuePilot/logs/`. TensorBoard log of the training run is in `~/ray_results/` 

## Contact
Please use GitHub's issues.