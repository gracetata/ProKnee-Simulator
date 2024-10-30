conda activate humanipro
MOTION="walk"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=oracle motion="$MOTION" wandb_activate=True