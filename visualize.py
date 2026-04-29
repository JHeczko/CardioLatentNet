import json
import os

import torch
from torch.utils.data import DataLoader
from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src import TransformerAec, LstmVae, CnnAec


def _run_analysis(model, run_name, model_info, test_loader, device):
    plots_dir = model_info["plots_dir"]
    x_hats, xs, ys = [], [], []
    latents = [] if hasattr(model, "encode") else None

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
            mse_sum          += torch.sum(diff ** 2).item()
            mae_sum          += torch.sum(torch.abs(diff)).item()
            signal_power_sum += torch.sum(x_batch_cpu ** 2).item()
            noise_power_sum  += torch.sum(diff ** 2).item()
            total_elements   += diff.numel()

            if latents is not None:
                latents.append(model.encode(x_batch).cpu())

    x_hat = torch.cat(x_hats, dim=0)
    x     = torch.cat(xs, dim=0)
    y     = torch.cat(ys, dim=0)

    mse  = mse_sum / total_elements
    mae  = mae_sum / total_elements
    rmse = mse ** 0.5
    snr  = 10 * torch.log10(torch.tensor(signal_power_sum / (noise_power_sum + 1e-8))).item()

    print(f"\n===== {run_name} BENCHMARK =====")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"SNR  : {snr:.2f} dB")
    print(f"Sample recon error (first): {torch.mean(torch.abs(x_hat[0] - x[0])).item():.6f}")

    if latents is not None:
        latent = torch.cat(latents, dim=0)
        for method in ('umap', 'tsne', 'pca'):
            visualize_latents(
                latent, y,
                model_title=run_name,
                method=method,
                path=os.path.join(plots_dir, f"{run_name}_latent_{method}.png"),
            )
        print(f"Latent mean: {latent.mean().item():.4f}  std: {latent.std(dim=0).mean().item():.4f}")


def process_model(model_info, test_loader):
    os.makedirs(model_info["plots_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== LOAD HISTORY =====
    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist']}") as f:
        history = json.load(f)
    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist_val']}") as f:
        history_val = json.load(f)

    plot_training_history(
        history,
        history_val,
        model_title=model_info['name'],
        path=os.path.join(model_info["plots_dir"], f"{model_info['name']}_history.png"),
    )

    # ===== ANALIZUJEMY OBA CHECKPOINTY =====
    checkpoints_to_eval = [
        (model_info['ckpt'],      model_info['name']),
        (model_info['best_ckpt'], model_info['name'] + "_Best"),
    ]

    for ckpt_file, run_name in checkpoints_to_eval:
        m = model_info['cls'](config=model_info['config']).to(device)
        m.eval()
        ckpt_path = f"{model_info['trainer_cfg'].checkpoint_dir}/{ckpt_file}"
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        _run_analysis(m, run_name, model_info, test_loader, device)


if __name__ == "__main__":
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl/", mode="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    models = [
        {
            "name": "LSTM_VAE_ver1",
            "cls": LstmVae,
            "config": LstmVaeConfig(),
            "trainer_cfg": LstmTrainerConfig(checkpoint_dir="checkpoints_lstm_ver1"),
            "ckpt": "lstm_model.pt",
            "best_ckpt": "lstm_best.pt",
            "hist": "lstm_history.json",
            "hist_val": "lstm_history_val.json",
            "plots_dir": "plots/plots_lstm_ver1",
        },
        {
            "name": "CNN",
            "cls": CnnAec,
            "config": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(checkpoint_dir="checkpoints_cnn_ver1"),
            "ckpt": "cnn_model.pt",
            "best_ckpt": "cnn_best.pt",
            "hist": "cnn_history.json",
            "hist_val": "cnn_history_val.json",
            "plots_dir": "plots/plots_cnn_ver1",
        },
    ]

    for model_info in models:
        process_model(model_info, test_loader)