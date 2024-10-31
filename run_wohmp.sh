conda activate humanipro
MOTION="walk"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=oracle motion="$MOTION" hmp=false wandb_activate=True
MOTION="run"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=oracle motion="$MOTION" hmp=false wandb_activate=True
MOTION="dance"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=oracle motion="$MOTION" hmp=false wandb_activate=True
MOTION="gym"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" hmp=false wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=oracle motion="$MOTION" hmp=false wandb_activate=True