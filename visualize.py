import json
from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LSTMTrainerConfig, TransformerTrainerConfig
from src.utils.config.model import LSTMConfig, TransformerUAECConfig
from src import TransformerUAEC, LstmCnnAEC

if __name__ == "__main__":
    config_lstm = LSTMConfig()
    config_transformer = TransformerUAECConfig()

    config_lstm_trainer = LSTMTrainerConfig()
    config_transformer_trainer = TransformerTrainerConfig()

    model_lstm = LstmCnnAEC(config=config_lstm)
    model_transformer = TransformerUAEC(config=config_transformer)

    model_lstm.load_state_dict(torch.load(f"{config_lstm_trainer.checkpoint_dir}/lstm_model.pt"))
    model_transformer.load_state_dict(torch.load(f"{config_transformer_trainer.checkpoint_dir}/transformer_model.pt"))

    test_ds = Hearbeat_ECG_DataSet(path="./dataset/ptb_xl_test/", mode="test")
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    x,y = next(iter(test_loader))

    # ==== HISTORY ====
    with open(f"{config_lstm_trainer.checkpoint_dir}/lstm_history.json") as f:
        lstm_history = json.load(f)
    with open(f"{config_lstm_trainer.checkpoint_dir}/lstm_history_val.json") as f:
        lstm_history_val = json.load(f)


    plot_training_history(lstm_history, lstm_history_val, model_title="LSTM VAE")

    with open(f"{config_transformer_trainer.checkpoint_dir}/transformer_history.json") as f:
        transformer_history = json.load(f)
    with open(f"{config_transformer_trainer.checkpoint_dir}/transformer_history_val.json") as f:
        transformer_history_val = json.load(f)

    plot_training_history(transformer_history, transformer_history_val, model_title="Transformer")


    # ==== LATENT VECTORS ====
    latent_lstm = model_lstm.encode(x)
    visualize_latents(latent_lstm, y, model_title="LSTM VAE")

    latent_transformer = model_transformer.encode(x)
    visualize_latents(latent_transformer, y, model_title="Transformer")
