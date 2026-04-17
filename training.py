from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.utils.trainers import LstmVeaTrainer
from src.utils.config.trainer import LSTMTrainerConfig
from src.data import Hearbeat_ECG_DataSet

from src import LstmCnnAEC

from matplotlib import pyplot as plt

if __name__ == '__main__':
    print("Loading dataset...", end=' ')
    train_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/")
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_ds, shuffle=True, batch_size=len(test_ds))
    print(f"Done. \nLoader size: {len(train_loader)}")

    print("Starting training...")
    config = LSTMTrainerConfig(
        max_iters=1000,
        lr=1e-3,
        device='cpu'
    )


    vae = LstmCnnAEC(2, 20, 60, 12, 0.2)
    #
    trainer = LstmVeaTrainer(
        model=vae,
        dataloader=train_loader,
        config=config
    )

    trainer.train()

    latent = None
    for x_all, _ in test_loader:
        latent = vae.encode(x_all)

    latents = latent.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3, s=5)
    plt.title("Latent space t-SNE")
    plt.show()

