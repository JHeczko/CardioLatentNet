from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.utils.trainers import LstmVeaTrainer
from src.utils.config.trainer import LSTMTrainerConfig
from src.utils.config.model import LSTMConfig
from src.data import Hearbeat_ECG_DataSet

from src import LstmCnnAEC

def train_lstm(train_ds, val_ds):
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=16)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=128)

    config_trainer = LSTMTrainerConfig(
        max_iters=1000,
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

    print("Starting training...")
    trainer.train()


def train_transformer(train_ds, val_ds): pass

def train_cnn(train_ds, val_ds): pass

if __name__ == '__main__':
    print("Loading dataset...", end=' ')

    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='train')
    val_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode='val')
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=16)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=128)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=len(test_ds))
    print(f"Done. \nLoader size: {len(train_loader)}")

    train_lstm(train_ds, val_ds)