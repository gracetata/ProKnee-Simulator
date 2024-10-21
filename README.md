# HumaniPro

## SetUp

```
conda create -n humanipro python==3.8
conda activate humanipro
```

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment
to simplify set up.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples`
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

```
pip install -r requirements.txt
```

## Train

```
python train.py task=HumanoidAMP
```

Register for [wandb](https://wandb.ai) and create a project named isaacgymenvs in wandb. Then, you can visualize the training process in the cloud using the following parameters.

```
wandb login
python train.py task=HumanoidAMP wandb_activate=True
```

## Test

```
python train.py task=HumanoidAMP checkpoint=checkpoints/walk_amp.pth test=True num_envs=1
```

## HumaniPro

```
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl wandb_activate=True
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=sarl checkpoint=**.pth
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=marl checkpoint=**.pth
```

Due to the complex dynamics of multi-body games, modeling it as a single-agent system leads to gradient explosion, where a learning rate around 8e-8 can succeed, but increasing it results in failure.

I increase the fps of simulator, which dt become 1/180s from 1/60s.

怀疑是此处的PPO代码在计算target_value时使用了当前的critic网络来计算batch中state的value， 因此导致值估计越推越高；
将代码改为在replay buffer中存入记录的同时存入state的值估计，而不是在计算target_value时计算state的值估计.

1. `--mode=sarl`: $\hat{A}(s.a) = \hat{A}^p_{\text{GAE}}(s, a^p).$ 
   1. `checkpoint=runs/HumanoidAMP_29-22-25-42/nn/HumanoidAMP_29-22-25-55_5000.pth` lr = 8.0e-08
   2. `21-00-00-30` lr = 8.0e-8
2. `--mode=marl`: $\hat{A}(s.a) = \hat{A}_{\text{GAE}}(s, a)$.
   1. `21-00-01-37` lr = 8.0e-8
