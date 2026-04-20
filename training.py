from torch.utils.data import DataLoader
from src.utils.trainers import LstmVeaTrainer, TransformerUAECTrainer, CnnAECTrainer
from src.utils.config.trainer import LSTMTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LSTMConfig, TransformerUAECConfig, CnnAECConfig
from src.data import Hearbeat_ECG_DataSet
from src import LstmCnnAEC, TransformerUAEC, CnnAEC


def run_training(train_ds, val_ds, test_ds, model_cls, trainer_cls, model_cfg, trainer_cfg, batch_sizes,
                 checkpoint_name, resume_training=False):
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_sizes['train'])
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_sizes['val'])
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=len(test_ds))

    model = model_cls(config=model_cfg)
    trainer = trainer_cls(model=model, dataloader=train_loader, val_dataloader=val_loader, config=trainer_cfg)

    if resume_training:
        trainer.load_checkpoint(f"{trainer_cfg.checkpoint_dir}/{checkpoint_name}")

    trainer.train()
    trainer.test(test_loader)

    return model


if __name__ == '__main__':
    print("Loading dataset...", end=' ')
    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='train')
    val_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='val')
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")
    print("Done")

    configs = [
        {
            "name": "CNN",
            "model_cls": CnnAEC,
            "trainer_cls": CnnAECTrainer,
            "model_cfg": CnnAECConfig(),
            "trainer_cfg": CnnTrainerConfig(max_iters=300, checkpoint_every=100, eval_every=100, device='cpu'),
            "batch_sizes": {'train': 16, 'val': 512},
            "ckpt": "cnn_newest.pt"
        },
        {
            "name": "TRANSFORMER",
            "model_cls": TransformerUAEC,
            "trainer_cls": TransformerUAECTrainer,
            "model_cfg": TransformerUAECConfig(),
            "trainer_cfg": TransformerTrainerConfig(max_iters=300, checkpoint_every=100, eval_every=100, device='cpu'),
            "batch_sizes": {'train': 32, 'val': 512},
            "ckpt": "transformer_newest.pt"
        },
        {
            "name": "LSTM VAE",
            "model_cls": LstmCnnAEC,
            "trainer_cls": LstmVeaTrainer,
            "model_cfg": LSTMConfig(),
            "trainer_cfg": LSTMTrainerConfig(max_iters=300, checkpoint_every=100, eval_every=100, device='cpu'),
            "batch_sizes": {'train': 32, 'val': 512},
            "ckpt": "lstm_newest.pt"
        },
    ]

    for cfg in configs:
        print(f"===== {cfg['name']} =====")
        run_training(
            train_ds, val_ds, test_ds,
            model_cls=cfg['model_cls'],
            trainer_cls=cfg['trainer_cls'],
            model_cfg=cfg['model_cfg'],
            trainer_cfg=cfg['trainer_cfg'],
            batch_sizes=cfg['batch_sizes'],
            checkpoint_name=cfg['ckpt']
        )