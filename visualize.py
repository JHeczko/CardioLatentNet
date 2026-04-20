import json
import torch
from torch.utils.data import DataLoader
from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src import TransformerAec, LstmVae, CnnAec


def process_model(model_info, x, y):
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

    plot_training_history(history, history_val, model_title=model_info['name'])

    # ===== MOVE DATA =====
    x = x.to(device)

    # ===== FORWARD =====
    with torch.no_grad():
        out = model(x)

        # handle VAE vs AEC
        if isinstance(out, tuple):
            x_hat = out[0]
        else:
            x_hat = out

    # ===== METRICS =====
    mse = torch.mean((x_hat - x) ** 2).item()
    mae = torch.mean(torch.abs(x_hat - x)).item()
    rmse = mse ** 0.5

    # SNR (Signal-to-Noise Ratio)
    signal_power = torch.mean(x ** 2)
    noise_power = torch.mean((x - x_hat) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()

    print(f"\n===== {model_info['name']} BENCHMARK =====")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"SNR  : {snr:.2f} dB")

    # ===== LATENT VISUALIZATION =====
    if hasattr(model, "encode"):
        with torch.no_grad():
            latent = model.encode(x)
        visualize_latents(latent.cpu(), y, model_title=model_info['name'], method='umap')

        # ===== LATENT STATS (important for VAE) =====
        latent_std = latent.std(dim=0).mean().item()
        latent_mean = latent.mean().item()

        print("\nLatent stats:")
        print(f"Mean: {latent_mean:.4f}")
        print(f"Std : {latent_std:.4f}")

    # ===== QUICK RECON CHECK =====
    # print small sample diff (debug sanity)
    diff = torch.mean(torch.abs(x_hat[0] - x[0])).item()
    print(f"\nSample reconstruction error (first sample): {diff:.6f}")

if __name__ == "__main__":
    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    x, y = next(iter(test_loader))

    models = [
        {
            "name": "LSTM VAE",
            "cls": LstmVae,
            "config": LstmVaeConfig(),
            "trainer_cfg": LstmTrainerConfig(),
            "ckpt": "lstm_model.pt",
            "hist": "lstm_history.json",
            "hist_val": "lstm_history_val.json"
        },
        {
            "name": "Transformer",
            "cls": TransformerAec,
            "config": TransformerAecConfig(),
            "trainer_cfg": TransformerTrainerConfig(),
            "ckpt": "transformer_model.pt",
            "hist": "transformer_history.json",
            "hist_val": "transformer_history_val.json"
        },
        {
            "name": "CNN",
            "cls": CnnAec,
            "config": CnnAecConfig(),
            "trainer_cfg": CnnTrainerConfig(),
            "ckpt": "cnn_model.pt",
            "hist": "cnn_history.json",
            "hist_val": "cnn_history_val.json"
        }
    ]

    for model_info in models:
        process_model(model_info, x, y)