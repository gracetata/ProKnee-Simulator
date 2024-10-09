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
python train.py task=HumanoidAMP checkpoint=checkpoints/run_amp.pth test=True num_envs=1
python train.py task=HumanoidAMP checkpoint=checkpoints/walk_amp.pth test=True num_envs=1
```

## HumaniPro

```
python train.py task=HumanoidAMP train=HumanoidAMPPro wandb_activate=True
python train.py task=HumanoidAMP test=True num_envs=1 train=HumanoidAMPPro checkpoint=**.pth
```


Due to the complex dynamics of multi-body games, modeling it as a single-agent system leads to gradient explosion, where a learning rate around 8e-8 can succeed, but increasing it results in failure.