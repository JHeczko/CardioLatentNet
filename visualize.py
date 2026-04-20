import json
import torch
from torch.utils.data import DataLoader
from src.visualize import plot_training_history, visualize_latents
from src.data import Hearbeat_ECG_DataSet
from src.utils.config.trainer import LstmTrainerConfig, TransformerTrainerConfig, CnnTrainerConfig
from src.utils.config.model import LstmVaeConfig, TransformerAecConfig, CnnAecConfig
from src import TransformerAec, LstmVae, CnnAec


def process_model(model_info, x, y):
    model = model_info['cls'](config=model_info['config'])
    ckpt_path = f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['ckpt']}"
    model.load_state_dict(torch.load(ckpt_path))

    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist']}") as f:
        history = json.load(f)
    with open(f"{model_info['trainer_cfg'].checkpoint_dir}/{model_info['hist_val']}") as f:
        history_val = json.load(f)

    plot_training_history(history, history_val, model_title=model_info['name'])

    latent = model.encode(x)
    visualize_latents(latent, y, model_title=model_info['name'], method='umap')


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