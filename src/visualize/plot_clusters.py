import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from typing import Literal

def visualize_latents(latents, targets, method:Literal['tsne', 'umap', 'pca']="tsne", model_title="", path=None):
    # 1. Konwersja na numpy
    latents_np = latents.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    if method == "tsne":
        if len(latents_np) > 10_000:
            idx = np.random.choice(len(latents_np), 10_000, replace=False)
            latents_np = latents_np[idx]
            targets_np = targets_np[idx]
        reducer = TSNE(n_components=2, perplexity=50, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)

    latents_2d = reducer.fit_transform(latents_np)

    # 3. Przygotowanie etykiet (dla uproszczenia bierzemy pierwszą aktywną klasę lub łączenie)
    # Tutaj robimy prostą metodę: zamieniamy tensor na stringi typu "0_2", "3_4"
    labels = []
    for row in targets_np:
        # Znajdujemy indeksy, gdzie jest 1
        active_classes = np.where(row == 1)[0]
        # Tworzymy stringa typu "c3_c4"
        labels.append("_".join([str(c) for c in active_classes]))

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latents_2d[:, 0],
        y=latents_2d[:, 1],
        hue=labels,
        palette="viridis",
        s=60, alpha=0.7
    )

    plt.title(f"Latent Space Visualization for {model_title} ({method})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if path is not None:
        print("[INFO] Saving figure to", path)
        plt.savefig(path)

    if path is None:
        plt.show()

    plt.close()

