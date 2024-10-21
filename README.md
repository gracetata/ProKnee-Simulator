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

1. `--mode=sarl`: $\hat{A}(s.a) = \hat{A}^p_{\text{GAE}}(s, a^p).$ 
   1. `checkpoint=runs/HumanoidAMP_29-22-25-42/nn/HumanoidAMP_29-22-25-55_5000.pth` lr = 8.0e-08
   2. `python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=sarl checkpoint=runs/HumanoidAMP_21-17-21-51/nn/HumanoidAMP_21-17-22-02_150.pth`
2. `--mode=marl`: $\hat{A}(s.a) = \hat{A}_{\text{GAE}}(s, a)$.
   1. `python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=marl checkpoint=runs/HumanoidAMP_21-17-22-07/nn/HumanoidAMP_21-17-22-17_150.pth`

