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
from src.data import Hearbeat_ECG_DataSet, Full_ECG_DataSet
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
    checkpoints_path = "./checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    print("Downloading dataset...", end=" ")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    print(f"Done! Path is = {os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")}")

    ds_path = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    test_ds_path = "./dataset/ptb_xl_test/"

    print("Loading dataset...\n", end=' ')
    train_heartbeat_ds = Hearbeat_ECG_DataSet(path=ds_path, mode='train')
    val_heartbeat_ds = Hearbeat_ECG_DataSet(path=ds_path, mode='val')
    test_heartbeat_ds = Hearbeat_ECG_DataSet(path=ds_path, mode="test")

    train_full_ds = Full_ECG_DataSet(path=ds_path, mode='train')
    val_full_ds = Full_ECG_DataSet(path=ds_path, mode='val')
    test_full_ds = Full_ECG_DataSet(path=ds_path, mode='test')

    ds_map = {
        "heartbeat": (train_heartbeat_ds, val_heartbeat_ds, test_heartbeat_ds),
        "full_ecg": (train_full_ds, val_full_ds, test_full_ds),
    }

    print("Done")

    configs = []

    configs_per_heartbeat = [
        # ======== CNN CONFS ========
        {
            "name": "CNN-baseline",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(early_stopper_patience=15, checkpoint_dir="checkpoints_cnn_baseline"),
            "batch_sizes": {'train': 128, 'val': 512},
            "type": "heartbeat",
            "resume_training": False,
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
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
            "type": "heartbeat",
            "resume_training": False
        },
    ]

    configs_per_heartbeat_plus = [
        # ======== CNN ========
        {
            "name": "CNN-HB-deeper",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                hidden_channels=128,
                latent_dim=64,
                blocks=4,
                dropout=0.15
            ),
            "trainer_cfg": CnnTrainerConfig(
                lr=1e-3,
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_cnn_deeper")
            ),
            "batch_sizes": {"train": 128, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "CNN-HB-wide-latent",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                latent_dim=128,
                blocks=4,
                dropout=0.15
            ),
            "trainer_cfg": CnnTrainerConfig(
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_cnn_wide_latent")
            ),
            "batch_sizes": {"train": 128, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "CNN-HB-deep-wide",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                hidden_channels=192,
                latent_dim=128,
                blocks=5,
                dropout=0.2
            ),
            "trainer_cfg": CnnTrainerConfig(
                lr=8e-4,
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_cnn_deep_wide")
            ),
            "batch_sizes": {"train": 128, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },

        # ======== LSTM VAE ========
        {
            "name": "LSTM-VAE-HB-big-boy-deeper",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=5,
                latent_dim=128,
                starting_channel_size=64,
                dropout=0.25
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1.5e-4,
                warmup_iters=5000,
                mmd_weight=0.2,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_lstm_big_boy_deeper")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "LSTM-VAE-HB-big-high-reg",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=4,
                latent_dim=128,
                starting_channel_size=64,
                dropout=0.35
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1.5e-4,
                warmup_iters=5000,
                mmd_weight=0.3,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_lstm_big_high_reg")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "LSTM-VAE-HB-big-wide-latent",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=4,
                latent_dim=192,
                starting_channel_size=96,
                dropout=0.25
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1e-4,
                warmup_iters=5000,
                mmd_weight=0.2,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_lstm_big_wide_latent")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "LSTM-VAE-HB-ultra-deeper-wider-latent",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=5,
                enc_dec_ratio=(2, 2),
                latent_dim=192,
                dropout=0.3
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1e-4,
                warmup_iters=5000,
                mmd_weight=0.2,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_lstm_ultra_deeper_wider_latent")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "LSTM-VAE-HB-deeper-high-reg",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                blocks=6,
                latent_dim=192,
                dropout=0.3
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1e-4,
                warmup_iters=5000,
                mmd_weight=0.8,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_lstm_deeper_high_reg")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },

        # ======== TRANSFORMER ========
        {
            "name": "TRANSFORMER-HB-big-deeper",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=4,
                enc_dec_ratio=(2, 2),
                hidden_dim=256,
                num_att_heads=8,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                early_stopper_patience=25,
                accumulation_step=2,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_transformer_big_deeper")
            ),
            "batch_sizes": {"train": 64, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "TRANSFORMER-HB-big-deeper-wider-latent",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=4,
                enc_dec_ratio=(2, 2),
                hidden_dim=256,
                num_att_heads=8,
                latent_dim=192,
                dropout=0.25,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                early_stopper_patience=25,
                accumulation_step=2,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_transformer_big_deeper_wider_latent")
            ),
            "batch_sizes": {"train": 64, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "TRANSFORMER-HB-big-wide-ratio",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=3,
                enc_dec_ratio=(3, 3),
                hidden_dim=384,
                num_att_heads=8,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                early_stopper_patience=25,
                accumulation_step=3,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_transformer_big_wide_ratio")
            ),
            "batch_sizes": {"train": 32, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "TRANSFORMER-HB-big-wide-more-att",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=6,
                hidden_dim=384,
                num_att_heads=12,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                early_stopper_patience=25,
                accumulation_step=5,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_transformer_big_wide_more_att")
            ),
            "batch_sizes": {"train": 16, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
        {
            "name": "TRANSFORMER-HB-big-wide-wider-latent",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                blocks=6,
                hidden_dim=384,
                num_att_heads=8,
                latent_dim=192,  # fix: było 196
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=2e-4,
                early_stopper_patience=25,
                checkpoint_dir=os.path.join(checkpoints_path, "hb_transformer_big_wide_wider_latent")
            ),
            "batch_sizes": {"train": 96, "val": 512},
            "type": "heartbeat",
            "resume_training": False,
        },
    ]

    configs_full_ecg = [

        # ======== CNN ========
        {
            "name": "FULL-CNN-deep",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                seq_len=1000,
                input_dim=12,
                hidden_channels=128,
                latent_dim=128,
                blocks=5,
                enc_dec_ratio=(2, 2),
                dropout=0.2
            ),
            "trainer_cfg": CnnTrainerConfig(
                lr=8e-4,
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path, "full_cnn_deep")
            ),
            "batch_sizes": {"train": 128, "val": 512},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-CNN-deeper",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(
                seq_len=1000,
                input_dim=12,
                hidden_channels=192,
                latent_dim=128,
                blocks=6,
                enc_dec_ratio=(2, 2),
                dropout=0.2
            ),
            "trainer_cfg": CnnTrainerConfig(
                lr=8e-4,
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path, "full_cnn_deeper")
            ),
            "batch_sizes": {"train": 128, "val": 512},
            "type": "full_ecg",
            "resume_training": False,
        },

        # ======== LSTM VAE ========
        {
            "name": "FULL-LSTM-deep",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                seq_len=1000,
                ecg_channels=12,
                blocks=5,
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=1e-4,
                warmup_iters=5000,
                mmd_weight=0.2,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_lstm_deep")
            ),
            "batch_sizes": {"train": 64, "val": 256},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-LSTM-big",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                seq_len=1000,
                ecg_channels=12,
                blocks=5,
                latent_dim=192,
                starting_channel_size=96,
                dropout=0.3
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=8e-5,
                warmup_iters=8000,
                max_iters=150_000,
                mmd_weight=0.2,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_lstm_big")
            ),
            "batch_sizes": {"train": 48, "val": 256},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-LSTM-big-ratio",
            "model_cls": LstmVae,
            "trainer_cls": LstmVaeTrainer,
            "model_cfg": LstmVaeConfig(
                seq_len=1000,
                ecg_channels=12,
                blocks=5,
                enc_dec_ratio=(2, 2),
                latent_dim=192,
                starting_channel_size=96,
                dropout=0.3
            ),
            "trainer_cfg": LstmTrainerConfig(
                lr=8e-5,
                warmup_iters=8000,
                max_iters=150_000,
                mmd_weight=0.2,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_lstm_big_ratio")
            ),
            "batch_sizes": {"train": 48, "val": 256},
            "type": "full_ecg",
            "resume_training": False,
        },

        # ======== TRANSFORMER ========
        {
            "name": "FULL-TRANSFORMER-baseline",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                seq_len=1000,
                input_dim=12,
                blocks=4,
                enc_dec_ratio=(2, 2),
                hidden_dim=256,
                num_att_heads=8,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=1e-4,
                weight_decay=1e-4,
                accumulation_step=6,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_transformer_baseline")
            ),
            "batch_sizes": {"train": 16, "val": 128},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-TRANSFORMER-deep",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                seq_len=1000,
                input_dim=12,
                blocks=6,
                enc_dec_ratio=(2, 2),
                hidden_dim=256,
                num_att_heads=8,
                latent_dim=128,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=8e-5,
                weight_decay=1e-4,
                warmup_iters=6000,
                accumulation_step=8,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_transformer_deep")
            ),
            "batch_sizes": {"train": 16, "val": 128},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-TRANSFORMER-big-deep",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                seq_len=1000,
                input_dim=12,
                blocks=6,
                enc_dec_ratio=(3, 3),
                hidden_dim=512,
                num_att_heads=8,
                latent_dim=256,
                dropout=0.25,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=8e-5,
                weight_decay=1e-4,
                warmup_iters=8000,
                max_iters=200_000,
                accumulation_step=32,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_transformer_big_deep")
            ),
            "batch_sizes": {"train": 4, "val": 128},
            "type": "full_ecg",
            "resume_training": False,
        },
        {
            "name": "FULL-TRANSFORMER-more-att",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(
                seq_len=1000,
                input_dim=12,
                blocks=5,
                enc_dec_ratio=(2, 2),
                hidden_dim=384,
                num_att_heads=12,
                latent_dim=192,
                dropout=0.2,
            ),
            "trainer_cfg": TransformerTrainerConfig(
                lr=8e-5,
                weight_decay=1e-4,
                warmup_iters=6000,
                accumulation_step=16,
                early_stopper_patience=20,
                checkpoint_dir=os.path.join(checkpoints_path, "full_transformer_more_att")
            ),
            "batch_sizes": {"train": 8, "val": 128},
            "type": "full_ecg",
            "resume_training": False,
        },
    ]

    configs.extend(configs_per_heartbeat_plus)
    configs.extend(configs_full_ecg)

    for i,cfg in enumerate(configs):

        # dataSet selection
        if cfg['type'] not in ds_map:
            raise TypeError(f"Unknown ds type: {cfg['type']}")
        train_ds, val_ds, test_ds = ds_map[cfg['type']]

        print(f"===== {cfg['name']} {i+1}/{len(configs)} =====")

        save_experiment_config(cfg, cfg["trainer_cfg"].checkpoint_dir)
        result = run_training(
            train_ds, val_ds, test_ds,
            model_cls=cfg['model_cls'],
            trainer_cls=cfg['trainer_cls'],
            model_cfg=cfg['model_cfg'],
            trainer_cfg=cfg['trainer_cfg'],
            batch_sizes=cfg['batch_sizes'],
            resume_training=cfg['resume_training'],
        )

        if result is None:
            print(f"===== ERROR FOR {cfg['name']} =====")

        print(f"======== END OF TRAINING {cfg['name']} =====")