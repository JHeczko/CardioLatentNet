import os.path
import os
import json
from dataclasses import asdict
from datetime import datetime

from torch.utils.data import DataLoader
from src.utils.trainers import LstmVaeTrainer, TransformerAecTrainer, CnnAecTrainer
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src.data import Hearbeat_ECG_DataSet
from src import LstmVae, TransformerAec, CnnAec

def run_training(train_ds, val_ds, test_ds, model_cls, trainer_cls, model_cfg, trainer_cfg, batch_sizes,
                 checkpoint_name = None, resume_training=False):
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_sizes['train'], pin_memory=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_sizes['val'], pin_memory=True, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=len(test_ds), pin_memory=True, num_workers=8, persistent_workers=True)

    model = model_cls(config=model_cfg)
    trainer = trainer_cls(model=model, dataloader=train_loader, val_dataloader=val_loader, config=trainer_cfg)

    if resume_training:
        trainer.load_checkpoint()

    trainer.train()
    trainer.test(test_loader)

    return model



def save_experiment_config(cfg, checkpoint_dir):
    """
    Saves full experiment configuration (model + trainer) into a single JSON file.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dump = {
        "timestamp": datetime.now().isoformat(),

        "model_cfg": (
            asdict(cfg["model_cfg"])
            if hasattr(cfg["model_cfg"], "__dataclass_fields__")
            else dict(cfg["model_cfg"])
        ),

        "trainer_cfg": (
            asdict(cfg["trainer_cfg"])
            if hasattr(cfg["trainer_cfg"], "__dataclass_fields__")
            else dict(cfg["trainer_cfg"])
        )
    }

    path = os.path.join(checkpoint_dir, "experiment_config.json")

    with open(path, "w") as f:
        json.dump(config_dump, f, indent=4)

    print(f"[CONFIG SAVED] -> {path}")

if __name__ == '__main__':
    print("Loading dataset...\n", end=' ')
    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='train')
    val_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='val')
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")
    print("Done")


    configs = [
        {
            "name": "CNN",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(early_stopper_patience=15),
            "batch_sizes": {'train': 128, 'val': 512},
            "ckpt": "cnn_newest.pt",
            "resume_training": False
        },
        {
            "name": "TRANSFORMER",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(),
            "trainer_cfg": TransformerTrainerConfig(),
            "batch_sizes": {'train': 128, 'val': 512},
            "ckpt": "transformer_newest.pt",
            "resume_training": False
        },
        {
            "name": "LSTM VAE",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(latent_dim=64),
            "trainer_cfg": LstmTrainerConfig(mmd_weight=0.7),
            "batch_sizes": {'train': 128, 'val': 512},
            "ckpt": "lstm_newest.pt",
            "resume_training": False
        },
    ]

    for cfg in configs:
        print(f"===== {cfg['name']} =====")

        save_experiment_config(cfg, cfg["trainer_cfg"].checkpoint_dir)
        run_training(
            train_ds, val_ds, test_ds,
            model_cls=cfg['model_cls'],
            trainer_cls=cfg['trainer_cls'],
            model_cfg=cfg['model_cfg'],
            trainer_cfg=cfg['trainer_cfg'],
            batch_sizes=cfg['batch_sizes'],
            #checkpoint_name=cfg['ckpt'],
            resume_training=cfg['resume_training']
        )