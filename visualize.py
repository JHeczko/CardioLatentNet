import json
import torch
from torch.utils.data import DataLoader
from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src import TransformerAec, LstmVae, CnnAec


def process_model(model_info, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_info['cls'](config=model_info['config']).to(device)
    model.eval()

    ckpt_path = f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['ckpt']}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ===== LOAD HISTORY =====
    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist']}") as f:
        history = json.load(f)
    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist_val']}") as f:
        history_val = json.load(f)

    plot_training_history(
        history,
        history_val,
        model_title=model_info['name'],
        path=f"{model_info['name']}_history.png"
    )

    # ===== INIT =====
    x_hats = []
    xs = []
    ys = []
    latents = [] if hasattr(model, "encode") else None

    mse_sum = 0.0
    mae_sum = 0.0
    signal_power_sum = 0.0
    noise_power_sum = 0.0
    total_elements = 0

    # ===== MAIN LOOP (single pass) =====
    with torch.no_grad():
        for batch in test_loader:

            # handle (x, y)
            if isinstance(batch, (list, tuple)):
                x_batch, y_batch = batch
                xs.append(x_batch)
                ys.append(y_batch)
            else:
                x_batch = batch

            x_batch = x_batch.to(device)

            # ===== FORWARD =====
            out = model(x_batch)

            if isinstance(out, tuple):  # VAE
                x_hat_batch = out[0]
            else:
                x_hat_batch = out

            x_hat_batch = x_hat_batch.cpu()
            x_batch_cpu = x_batch.cpu()

            x_hats.append(x_hat_batch)

            # ===== METRICS (streaming) =====
            diff = x_hat_batch - x_batch_cpu

            mse_sum += torch.sum(diff ** 2).item()
            mae_sum += torch.sum(torch.abs(diff)).item()

            signal_power_sum += torch.sum(x_batch_cpu ** 2).item()
            noise_power_sum += torch.sum(diff ** 2).item()

            total_elements += diff.numel()

            # ===== LATENT =====
            if latents is not None:
                z = model.encode(x_batch)
                latents.append(z.cpu())

    # ===== FINAL TENSORS =====
    x_hat = torch.cat(x_hats, dim=0)
    y = torch.cat(ys, dim=0)
    x = torch.cat(xs, dim=0)

    # ===== FINAL METRICS =====
    mse = mse_sum / total_elements
    mae = mae_sum / total_elements
    rmse = mse ** 0.5
    snr = 10 * torch.log10(torch.tensor(signal_power_sum / (noise_power_sum + 1e-8))).item()

    print(f"\n===== {model_info['name']} BENCHMARK =====")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"SNR  : {snr:.2f} dB")

    # ===== LATENT VIS =====
    if latents is not None:
        latent = torch.cat(latents, dim=0)

        visualize_latents(latent, y, model_title=model_info['name'], method='umap',
                          path=f"{model_info['name']}_latent_umap.png")
        visualize_latents(latent, y, model_title=model_info['name'], method='tsne',
                          path=f"{model_info['name']}_latent_tsne.png")
        visualize_latents(latent, y, model_title=model_info['name'], method='pca',
                          path=f"{model_info['name']}_latent_pca.png")

        # ===== LATENT STATS =====
        latent_std = latent.std(dim=0).mean().item()
        latent_mean = latent.mean().item()

        print("\nLatent stats:")
        print(f"Mean: {latent_mean:.4f}")
        print(f"Std : {latent_std:.4f}")

    # ===== QUICK RECON CHECK =====
    diff_sample = torch.mean(torch.abs(x_hat[0] - x[0])).item()
    print(f"\nSample reconstruction error (first sample): {diff_sample:.6f}")

if __name__ == "__main__":
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl/", mode="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    models = [
        {
            "name": "LSTM_VAE",
            "cls": LstmVae,
            "config": LstmVaeConfig(),
            "trainer_cfg": LstmTrainerConfig(checkpoint_dir="checkpoints_lstm_ver1"),
            "ckpt": "lstm_model.pt",
            "hist": "lstm_history.json",
            "hist_val": "lstm_history_val.json"
        },
        {
            "name": "LSTM_VAE_Best_Model",
            "cls": LstmVae,
            "config": LstmVaeConfig(),
            "trainer_cfg": LstmTrainerConfig(checkpoint_dir="checkpoints_lstm_ver1"),
            "ckpt": "lstm_best.pt",
            "hist": "lstm_history.json",
            "hist_val": "lstm_history_val.json"
        }
        # {
        #     "name": "Transformer",
        #     "cls": TransformerAec,
        #     "config": TransformerAecConfig(),
        #     "trainer_cfg": TransformerTrainerConfig(),
        #     "ckpt": "transformer_model.pt",
        #     "hist": "transformer_history.json",
        #     "hist_val": "transformer_history_val.json"
        # },
        # {
        #     "name": "CNN",
        #     "cls": CnnAec,
        #     "config": CnnAecConfig(),
        #     "trainer_cfg": CnnTrainerConfig(),
        #     "ckpt": "cnn_model.pt",
        #     "hist": "cnn_history.json",
        #     "hist_val": "cnn_history_val.json"
        # }
    ]

    for model_info in models:
        process_model(model_info, test_loader)