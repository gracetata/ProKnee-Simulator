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

FPS: 1/180s for all task.

## Human Model

### Train

```
python train.py task=HumanoidAMP
```

Register for [wandb](https://wandb.ai) and create a project named isaacgymenvs in wandb. Then, you can visualize the training process in the cloud using the following parameters.

```
wandb login
python train.py task=HumanoidAMP wandb_activate=True
```

### Test

```
python train.py task=HumanoidAMP checkpoint=checkpoints/walk_amp.pth test=True num_envs=1
```

## HumaniPro

### Single-Agent RL (SARL, PPO)

$\hat{A}(s.a) = \hat{A}^p_{\text{GAE}}(s, a^p).$ 

```
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl wandb_activate=True
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=sarl checkpoint=**.pth
```

Results:
- Bad result (Walk, lr = $8.0 \times 10^{-8}$)
```
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=sarl checkpoint=runs/HumanoidAMP_29-22-25-42/nn/HumanoidAMP_29-22-25-55_5000.pth
```
- Good result (Walk, lr = $5.0 \times 10^{-5}$)
```
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=sarl checkpoint=runs/HumanoidAMP_21-17-21-51/nn/HumanoidAMP_21-17-22-02_150.pth
```

### Multi-Agent RL (MARL, ours)

$\hat{A}(s.a) = \hat{A}_{\text{GAE}}(s, a)$.

```
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl wandb_activate=True
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=marl checkpoint=**.pth
```

Results:
- Good result (Walk, lr = $5.0 \times 10^{-5}$)
```
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro mode=marl checkpoint=runs/HumanoidAMP_21-17-22-07/nn/HumanoidAMP_21-17-22-17_150.pth
```

