from torch.utils.data import DataLoader
from src.utils.trainers import LstmVeaTrainer, TransformerUAECTrainer
from src.utils.config.trainer import LSTMTrainerConfig, TransformerTrainerConfig
from src.utils.config.model import LSTMConfig, TransformerUAECConfig
from src.data import Hearbeat_ECG_DataSet

from src import LstmCnnAEC,TransformerUAEC

def train_lstm(train_ds, val_ds, test_ds, resume_training=False):
    print("===== LSTM VAE =====")

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=32)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=512)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=len(test_ds))

    config_trainer = LSTMTrainerConfig(
        max_iters=300,
        checkpoint_every=100,
        eval_every=100,
        device='cpu'
    )
    config_model = LSTMConfig()

    vae = LstmCnnAEC(config = config_model)

    trainer = LstmVeaTrainer(
        model=vae,
        dataloader=train_loader,
        val_dataloader=val_loader,
        config=config_trainer
    )

    if resume_training:
        print(f"Loading the checkpoint({config_trainer.checkpoint_dir}/lstm_newest.pt)...")
        trainer.load_checkpoint(f"{config_trainer.checkpoint_dir}/lstm_newest.pt")

    print("Starting training...")
    trainer.train()

    print("Done training...")
    trainer.test(test_loader)

    return vae

def train_transformer(train_ds, val_ds, test_ds, resume_training=False):
    print("===== TRANSFORMER =====")

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=16)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=128)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=len(test_ds))

    config_trainer = TransformerTrainerConfig(
        max_iters=300,
        checkpoint_every=100,
        eval_every=100,
        device='cpu'
    )
    config_model = TransformerUAECConfig()

    uaec = TransformerUAEC(config = config_model)

    trainer = TransformerUAECTrainer(
        model=uaec,
        dataloader=train_loader,
        val_dataloader=val_loader,
        config=config_trainer
    )

    if resume_training:
        print(f"Loading the checkpoint({config_trainer.checkpoint_dir}/transformer_newest.pt)...")
        trainer.load_checkpoint(f"{config_trainer.checkpoint_dir}/transformer_newest.pt")

    print("Starting training...")
    trainer.train()

    print("Done training...")
    trainer.test(test_loader)

    return uaec

def train_cnn(train_ds, val_ds, test_ds, resume_training=False): pass

if __name__ == '__main__':
    print("Loading dataset...", end=' ')

    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='train')
    val_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='val')
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")

    print("Done")

    model_transformer = train_transformer(train_ds, val_ds, test_ds)
    model_vae = train_lstm(train_ds, val_ds, test_ds)