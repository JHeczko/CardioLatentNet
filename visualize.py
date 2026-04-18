from src.utils.config.trainer import LSTMTrainerConfig

if __name__ == "__main__":
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