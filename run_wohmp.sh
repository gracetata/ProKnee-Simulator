conda activate humanipro
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion=walk wandb_activate=True hmp=false
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion=run wandb_activate=True hmp=false
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion=dance wandb_activate=True hmp=false
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion=gym wandb_activate=True hmp=false