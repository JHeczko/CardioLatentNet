from torch.utils.data import DataLoader
from src.utils.trainers import LSTMCnnVAETrainer
from src.utils.config.lstm import TrainerConfig
from src.data import Hearbeat_ECG_DataSet

from src import LstmCnnAEC

if __name__ == '__main__':
    print("Loading dataset...", end=' ')
    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/")
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=16)
    print(f"Done. \nLoader size: {len(train_loader)}")

    print("Starting training...")
    config = TrainerConfig(
        max_iters=50000,
        lr=1e-3,
        beta_warmup_iters=5000,
        device='cpu'
    )

    vae = LstmCnnAEC(2, 20, 60, 12, 0.2)

    trainer = LSTMCnnVAETrainer(
        model=vae,
        dataloader=train_loader,
        config=config
    )

    trainer.train()