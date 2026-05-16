import json
import os
import pickle
import warnings


import kagglehub

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

from src.utils.trainers import CnnAecTrainer, LstmVaeTrainer, TransformerAecTrainer
from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src import TransformerAec, LstmVae, CnnAec


def get_model_prefix(model_cls):
    """
    Wyciąga prefix modelu (cnn, transformer, lstm) na podstawie klasy,
    aby poprawnie mapować nazwy plików z checkpointami.
    """
    cls_name = model_cls.__name__.lower()
    if 'cnn' in cls_name:
        return 'cnn'
    elif 'transformer' in cls_name:
        return 'transformer'
    elif 'lstm' in cls_name:
        return 'lstm'
    else:
        raise ValueError(f"[ERROR] Nieznana klasa modelu: {cls_name}, nie można wywnioskować prefixu.")


def evaluate_latent_quality(latent_np, n_clusters=18):
    """Ocenia jakość przestrzeni latentnej niezależnie od rekonstrukcji."""

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_np)

    silhouette = silhouette_score(latent_np, cluster_labels, sample_size=10_000)
    davies_bouldin = davies_bouldin_score(latent_np, cluster_labels)

    # aktywne wymiary — ile dims ma std > próg
    stds = latent_np.std(axis=0)
    active_dims = int((stds > 0.1).sum())

    return {
        "silhouette": round(silhouette, 4),      # wyższy = lepszy, max 1.0
        "davies_bouldin": round(davies_bouldin, 4),  # niższy = lepszy
        "active_dims": active_dims,
    }

def _run_analysis(model, run_name, plots_dir, test_loader, device):
    result = {
        'name': run_name
    }

    x_hats, xs, ys = [], [], []
    latents = [] if hasattr(model, "encode") else None

    lines = []

    mse_sum = mae_sum = signal_power_sum = noise_power_sum = 0.0
    total_elements = 0

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x_batch, y_batch = batch
                xs.append(x_batch)
                ys.append(y_batch)
            else:
                x_batch = batch

            x_batch = x_batch.to(device)
            out = model(x_batch)
            x_hat_batch = (out[0] if isinstance(out, tuple) else out).cpu()
            x_batch_cpu = x_batch.cpu()
            x_hats.append(x_hat_batch)

            diff = x_hat_batch - x_batch_cpu
            mse_sum += torch.sum(diff ** 2).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            signal_power_sum += torch.sum(x_batch_cpu ** 2).item()
            noise_power_sum += torch.sum(diff ** 2).item()
            total_elements += diff.numel()

            if latents is not None:
                latents.append(model.encode(x_batch).cpu())

    x_hat = torch.cat(x_hats, dim=0)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    mse = mse_sum / total_elements
    mae = mae_sum / total_elements
    rmse = mse ** 0.5
    snr = 10 * torch.log10(torch.tensor(signal_power_sum / (noise_power_sum + 1e-8))).item()

    result['mse'] = mse
    result['mae'] = mae
    result['rmse'] = rmse
    result['snr'] = snr

    lines.append(f"\n===== {run_name} BENCHMARK =====")
    lines.append(f"MSE  : {mse:.6f}")
    lines.append(f"MAE  : {mae:.6f}")
    lines.append(f"RMSE  : {rmse:.6f}")
    lines.append(f"SNR  : {snr:.6f}")
    lines.append(f"Sample recon error (first): {torch.mean(torch.abs(x_hat[0] - x[0])).item():.6f}")

    print(f"\n===== {run_name} BENCHMARK =====")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"SNR  : {snr:.2f} dB")
    print(f"Sample recon error (first): {torch.mean(torch.abs(x_hat[0] - x[0])).item():.6f}")

    if latents is not None:
        latent = torch.cat(latents, dim=0)

        latent_metrics = evaluate_latent_quality(latent.numpy())

        for method in ('umap', 'tsne', 'pca'):
            visualize_latents(
                latent, y,
                model_title=run_name,
                method=method,
                path=os.path.join(plots_dir, f"{run_name}_latent_{method}.png"),
            )
        latent_mean = latent.mean().item()
        latent_std = latent.std(dim=0).mean().item()

        result['latent_mean'] = latent_mean
        result['latent_std'] = latent_std

        print(f"Latent mean: {latent_mean:.4f}  std: {latent_std:.4f}")
        lines.append(f"Latent mean: {latent_mean:.4f}  std: {latent_std:.4f}")

        result.update(latent_metrics)

        print("\n====== KMEANS SIMPLE CLUSTERING =====")
        print(f"Silhouette(kmeans, higher=better)         : {latent_metrics['silhouette']}")
        print(f"Davies Bouldin Score(kmeans, lower=better): {latent_metrics['davies_bouldin']}")
        print(f"Active Dimensions                         : {latent_metrics['active_dims']}")

        lines.append("\n====== KMEANS SIMPLE CLUSTERING =====\n")
        lines.append(f"Silhouette(kmeans, higher=better)         : {latent_metrics['silhouette']}")
        lines.append(f"Davies Bouldin Score(kmeans, lower=better): {latent_metrics['davies_bouldin']}")
        lines.append(f"Active Dimensions                         : {latent_metrics['active_dims']}")

        output = "\n".join(lines)
        # ===== ZAPIS DO PLIKU =====
        log_path = os.path.join(plots_dir, f"{run_name}_results.txt")
        with open(log_path, "w") as f:
            f.write(output)

    return result

def process_model(config, test_loader):
    run_name = config["name"]
    plots_dir = os.path.join("plots", run_name)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rozpakowujemy config
    model_cls = config["model_cls"]
    model_cfg = config["model_cfg"]
    trainer_cfg = config["trainer_cfg"]

    # Wyciągamy potrzebne ścieżki
    ckpt_dir = trainer_cfg.checkpoint_dir
    prefix = get_model_prefix(model_cls)  # np. 'lstm', 'cnn'

    hist_path = os.path.join(ckpt_dir, f"{prefix}_history.json")
    hist_val_path = os.path.join(ckpt_dir, f"{prefix}_history_val.json")

    # ===== LOAD HISTORY =====
    try:
        with open(hist_path, 'r') as f:
            history = json.load(f)
        with open(hist_val_path, 'r') as f:
            history_val = json.load(f)

        plot_training_history(
            history,
            history_val,
            model_title=run_name,
            path=os.path.join(plots_dir, f"{run_name}_history.png"),
        )
    except FileNotFoundError as e:
        print(f"[WARNING] ⚠️ Nie znaleziono plików historii dla {run_name}: {e}")

    # ===== ANALIZUJEMY OBA CHECKPOINTY =====
    checkpoints_to_eval = [
        (f"{prefix}_model.pt", run_name),  # Ostatni model
        (f"{prefix}_best.pt", f"{run_name}_Best"),  # Najlepszy model
    ]

    results = []
    for ckpt_file, eval_name in checkpoints_to_eval:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)

        if not os.path.exists(ckpt_path):
            print(f"[WARNING] ⚠️ Checkpoint nie istnieje, pomijam ewaluację: {ckpt_path}")
            continue

        m = model_cls(config=model_cfg).to(device)
        m.eval()

        state_dict = torch.load(ckpt_path, map_location=device)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        m.load_state_dict(state_dict)

        result = _run_analysis(m, eval_name, plots_dir, test_loader, device)
        results.append(result)

    return results

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch._C._jit_set_profiling_executor(False)

    print("Downloading dataset...", end=" ")
    path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
    print(f"Done! Path is = {os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")}")

    ds_path = os.path.join(path, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    test_ds_path = "./dataset/ptb_xl_test/"

    print("Loading dataset...\n", end=' ')
    test_ds = Hearbeat_ECG_DataSet(path=ds_path, mode="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    print("Done")

    checkpoints_path = "./checkpoints"

    # Tutaj dokładnie Twoja lista ze słownikami treningowymi (configs)
    configs = [
        # {
        #     "name": "LSTM_VAE_ver1",
        #     "model_cls": LstmVae,
        #     "trainer_cls": LstmVaeTrainer,
        #     "model_cfg": LstmVaeConfig(),
        #     "trainer_cfg": LstmTrainerConfig(
        #         checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_ver1")
        #     ),
        #     "batch_sizes": {'train': 128, 'val': 512},
        #     "resume_training": False
        # },
        # {
        #     "name": "CNN_AEC_ver1",
        #     "model_cls": CnnAec,
        #     "trainer_cls": CnnAecTrainer,
        #     "model_cfg": CnnAecConfig(),
        #     "trainer_cfg": CnnTrainerConfig(
        #         checkpoint_dir=os.path.join(checkpoints_path, "checkpoints/checkpoints_cnn_ver1")
        #     ),
        #     "batch_sizes": {'train': 128, 'val': 512},
        # },

        # remote machine training
        # ======== CNN CONFS ========
        {
            "name": "CNN-baseline",
            "model_cls": CnnAec,
            "trainer_cls": CnnAecTrainer,
            "model_cfg": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(
                early_stopper_patience=15,
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_cnn_baseline")
            ),
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_cnn_fast_and_stable")),
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_cnn_deep")),
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_cnn_hard_bottleneck")),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        },
            # ======= Transformer config =======
        {
            "name": "TRANSFORMER-baseline",
            "model_cls": TransformerAec,
            "trainer_cls": TransformerAecTrainer,
            "model_cfg": TransformerAecConfig(),
            "trainer_cfg": TransformerTrainerConfig(checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_transformer_baseline")),
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_transformer_stable_baseline")),
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_transformer_bigboy")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_transformer_no_reg")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_baseline")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_baseline_pp")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_strong_latent")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_big_boy")
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
                checkpoint_dir=os.path.join(checkpoints_path,"checkpoints_lstm_reg_latent")
            ),
            "batch_sizes": {'train': 128, 'val': 512},
            "resume_training": False
        }

        # from remote VM
    ]

    results = []
    for cfg in configs:
        print(f"\n======================================")
        print(f"Przetwarzanie modelu: {cfg['name']}")
        print(f"======================================")
        results_model = process_model(cfg, test_loader)
        results.extend(results_model)

    with open("./results.json", "w") as f:
        json.dump(results, f, indent=2)