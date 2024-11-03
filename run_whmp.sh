conda activate humanipro
MOTION="walk"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" wandb_activate=True
MOTION="run"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" wandb_activate=True
MOTION="dance"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" wandb_activate=True
MOTION="gym"
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=sarl motion="$MOTION" wandb_activate=True
python train.py task=HumanoidAMP train=HumanoidAMPPro mode=marl motion="$MOTION" wandb_activate=True