import gc
import os.path
import os
import json
import traceback
import kagglehub
from dataclasses import asdict
from datetime import datetime

import torch.cuda
from torch.utils.data import DataLoader

from src.utils.trainers import LstmVaeTrainer, TransformerAecTrainer, CnnAecTrainer
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src.data import Hearbeat_ECG_DataSet
from src import LstmVae, TransformerAec, CnnAec

def run_training(train_ds, val_ds, test_ds, model_cls, trainer_cls, model_cfg, trainer_cfg, batch_sizes, resume_training=False):
    try:
        cpu_cores = os.cpu_count() or 4
        num_workers = max(2, cpu_cores)

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_sizes['train'], pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_sizes['val'], pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_sizes['val'], pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2)

        model = model_cls(config=model_cfg)
        trainer = trainer_cls(model=model, dataloader=train_loader, val_dataloader=val_loader, config=trainer_cfg)

        if resume_training:
            trainer.load_checkpoint()

        trainer.train()
        trainer.test(test_loader)

        return model
    except Exception as e:
        print(f"\n[ERROR] Model {model_cls.__name__} wywalił się z błędem:")
        print(traceback.format_exc())

        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer

        torch.cuda.empty_cache()
        gc.collect()

        return None

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
    print("Downloading dataset...", end=" ")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    print(f"Done! Path is = {os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")}")

    ds_path = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    test_ds_path = "./dataset/ptb_xl_test/"

    print("Loading dataset...\n", end=' ')
    train_ds = Hearbeat_ECG_DataSet(path=ds_path, mode='train')
    val_ds = Hearbeat_ECG_DataSet(path=ds_path, mode='val')
    test_ds = Hearbeat_ECG_DataSet(path=ds_path, mode="test")
    print("Done")


    configs = [
        # ======== CNN CONFS ========
        {
            "name": "CNN-baseline",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(early_stopper_patience=15, checkpoint_dir="checkpoints_cnn_baseline"),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "CNN-fast&stable",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                hidden_channels=128,
                latent_dim=64,
                blocks=3,
                dropout=0.15
            ),
            "trainer_cfg": CnnTrainerConfig(
                early_stopper_patience=15,
                lr=1e-3,
                checkpoint_dir="checkpoints_cnn_fast_and_stable"),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "CNN-deep",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                hidden_channels=128,
                latent_dim=64,
                blocks=5,
                dropout=0.2
            ),
            "trainer_cfg": CnnTrainerConfig(
                early_stopper_patience=15,
                checkpoint_dir="checkpoints_cnn_deep"),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "CNN-hard-bottleneck",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                hidden_channels=128,
                latent_dim=16,
                blocks=3,
                dropout=0.2
            ),
            "trainer_cfg": CnnTrainerConfig(
                early_stopper_patience=15,
                checkpoint_dir="checkpoints_cnn_hard_bottleneck"),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        # ======= Transformer config =======
        {
            "name": "TRANSFORMER-baseline",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(),
            "trainer_cfg": TransformerTrainerConfig(checkpoint_dir="checkpoints_transformer_baseline"),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "TRANSFORMER-stable-baseline",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=4,
                hidden_dim=192,
                num_att_heads=6,
                latent_dim=64,
                dropout=0.15
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                weight_decay=1e-4,
                checkpoint_dir="checkpoints_transformer_stable_baseline"),
            "batch_sizes": {'train': 96, 'val': 512},
            "resume_training": False
        },
        {
            "name": "TRANSFORMER-big-boy",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=6,
                hidden_dim=256,
                num_att_heads=8,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                weight_decay=1e-4,
                checkpoint_dir="checkpoints_transformer_bigboy"
            ),
            "batch_sizes": {'train': 96, 'val': 512},
            "resume_training": False
        },
        {
            "name": "TRANSFORMER-no-reg",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=4,
                hidden_dim=256,
                latent_dim=64,
                dropout=0.05
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=1e-4,
                weight_decay=1e-5,
                checkpoint_dir="checkpoints_transformer_no_reg"
            ),
            "batch_sizes": {'train': 64, 'val': 512},
            "resume_training": False
        },
        # ======= LSTM VAE config =======
        {
            "name": "LSTM-VAE-baseline",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                latent_dim=64
            ),
            "trainer_cfg": LstmTrainerConfig(
                mmd_weight=0.7,
                checkpoint_dir="checkpoints_lstm_baseline"
            ),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "LSTM-VAE-baseline-pp",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=3,
                latent_dim=96,
                starting_channel_size=64,
                dropout=0.2
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=2e-4,
                warmup_iters=4000,
                mmd_weight=0.1,
                checkpoint_dir="checkpoints_lstm_baseline_pp"
             ),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "LSTM-VAE-strong-latent",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=3,
                latent_dim=64,
                starting_channel_size=64,
                dropout=0.2
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=2e-4,
                mmd_weight=0.5,
                checkpoint_dir="checkpoints_lstm_strong_latent"
            ),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
        {
            "name": "LSTM-VAE-big-model",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=4,
                latent_dim=128,
                starting_channel_size=64,
                dropout=0.25
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1.5e-4,
                warmup_iters=5000,
                mmd_weight=0.2,
                checkpoint_dir="checkpoints_lstm_big_boy"
            ),
            "batch_sizes": {'train': 96, 'val': 512},
            "resume_training": False
        },
        {
            "name": "LSTM-VAE-regularized-latent",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=3,
                latent_dim=32,
                starting_channel_size=64,
                dropout=0.3
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=2e-4,
                mmd_weight=1.0,
                checkpoint_dir="checkpoints_lstm_reg_latent"
            ),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
    ]

    for i,cfg in enumerate(configs):
        print(f"===== {cfg['name']} {i}/{len(cfg)} =====")

        save_experiment_config(cfg, cfg["trainer_cfg"].checkpoint_dir)
        result = run_training(
            train_ds, val_ds, test_ds,
            model_cls=cfg['model_cls'],
            trainer_cls=cfg['trainer_cls'],
            model_cfg=cfg['model_cfg'],
            trainer_cfg=cfg['trainer_cfg'],
            batch_sizes=cfg['batch_sizes'],
            resume_training=cfg['resume_training']
        )

        if result is None:
            print(f"===== ERROR FOR {cfg['name']} =====")

        print(f"======== END OF TRAINING {cfg['name']} =====")