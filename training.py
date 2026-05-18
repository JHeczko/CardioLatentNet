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

MODEL_REGISTRY = {
    "CnnAec": CnnAec,
    "LstmVae": LstmVae,
    "TransformerAec": TransformerAec,
}

TRAINER_REGISTRY = {
    "CnnAecTrainer": CnnAecTrainer,
    "LstmVaeTrainer": LstmVaeTrainer,
    "TransformerAecTrainer": TransformerAecTrainer,
}

MODEL_CFG_REGISTRY = {
    "CnnAec": CnnAecConfig,
    "LstmVae": LstmVaeConfig,
    "TransformerAec": TransformerAecConfig,
}

TRAINER_CFG_REGISTRY = {
    "CnnAecTrainer": CnnTrainerConfig,
    "LstmVaeTrainer": LstmTrainerConfig,
    "TransformerAecTrainer": TransformerTrainerConfig,
}

def run_training(train_ds, val_ds, test_ds, model_cls, trainer_cls, model_cfg, trainer_cfg, batch_sizes, resume_training=False):
    torch.cuda.empty_cache()
    gc.collect()

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

def load_configs(path: str, checkpoints_base: str):
    with open(path, "r") as f:
        raw = json.load(f)

    configs = []
    for item in raw:
        model_cls_name = item["model_cls"]
        trainer_cls_name = item["trainer_cls"]

        # podmień checkpoint_dir na absolutny
        item["trainer_cfg"]["checkpoint_dir"] = os.path.join(
            checkpoints_base, item["trainer_cfg"]["checkpoint_dir"]
        )

        configs.append({
            "name": item["name"],
            "model_cls": MODEL_REGISTRY[model_cls_name],
            "trainer_cls": TRAINER_REGISTRY[trainer_cls_name],
            "model_cfg": MODEL_CFG_REGISTRY[model_cls_name](**item["model_cfg"]),
            "trainer_cfg": TRAINER_CFG_REGISTRY[trainer_cls_name](**item["trainer_cfg"]),
            "batch_sizes": item["batch_sizes"],
            "type": item["type"],
            "resume_training": item.get("resume_training", False),
        })

    return configs

if __name__ == '__main__':
    checkpoints_base = "./checkpoints"

    checkpoints_full_path = os.path.join(checkpoints_base, "./checkpoints_full")
    checkpoints_heartbeat_path = os.path.join(checkpoints_base, "./checkpoints_heartbeat")

    os.makedirs(checkpoints_heartbeat_path, exist_ok=True)
    os.makedirs(checkpoints_full_path, exist_ok=True)

    print("=====\nDownloading dataset...")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    print(f"Done! Path is = {os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")}")

    ds_path = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    test_ds_path = "./dataset/ptb_xl_test/"

    print("\n\n=====\nLoading dataset...\n", end=' ')
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

    #configs += load_configs("./experiments/experiment_heartbeat.json", checkpoints_heartbeat_path)
    configs += load_configs("experiments/experiment_heartbeat2.json", checkpoints_heartbeat_path)
    configs += load_configs("./experiments/experiment_full.json", checkpoints_full_path)

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