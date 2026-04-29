# CardioLatentNet
## The goal of the project
Stworzenie autoenkodera dla szeregów EKG i analiza latent‑space (UMAP/t‑SNE) pod kątem lokalizacji jednostek chorobowych, płci i innych metadanych; w drugiej części – prosta klasyfikacja na wektorach latentnych.

- Technology: Python

## Dane i preprocessing:
Wybór publicznego zbioru (np. PTB‑XL) (dostępna wersja już po preprocesingu na łame chunk równej długości)

## Model
- Latent-space:
  - 1D‑CNN/T ransformer/LSTMConv autoencoder (rekonstrukcja sygnału)
  - ekstrakcja wektorów latentnych.
- Analiza latent‑space:
  - wizualizacja (UMAP/t‑SNE)
- Klasyfikacja/klasteryzacja (LR/SVM/MLP) na latentach dla: diagnoz, płci, wieku (jeśli dostępne).
- Ewaluacja:
  - rekonstrukcja (MSE/MAE), 
  - klasyfikacja (Accuracy/F1/AUROC lub w przypad), 
  - inspekcja „czytelności” latentów (separowalność klas).

## Checkpoints
All checkpoints are structured as follows({model} is the name of the model, and we are not talking about the name such as `LSTM-BASELINE-STABLE`, but the base name of the model, for example for all `LSTM` models trainer is going to use `{model}=lstm`, it is hardcoded into the trainer):
- `experiment_config.json`: containts all the parameters for model and trainer config, with all params used for training the specified in file model
- `{model}_history.json`: contains training history(constains every step of the model) 
- `{model}_history_val.json`: contains validation history(every `eval_every` specified in the config files)
- `{model}_best.pt`(best model weights): this is the best model by validation error in whole training(containts only model weight)
- `{model}_model.pt`(last model weight): this is the last saved model, this corresponds to the model that was evaluated at the last step of `{model}_history_val.json` 
- `{model}_newest.pt`(last checkpoint): this is proper training checkpoint. This is being saved every `checkpoint_every` specified in the trainer config. Checkpoints contains:
  - last model weights
  - optimizer params
  - patience counter and best val in whole training
  - last saved step
  - history of training
  - history of validation

## Deliverables
- notebooki 
- raport PDF
- wykresy latent‑space

[//]: # (- torch: dowolny )
[//]: # (- cuda: 11.8)
[//]: # (- python: 3.11)

## Links
- [NATURE](https://www.nature.com/articles/s41597-020-0495-6)
- [PHYSIONET](https://physionet.org/content/ptb-xl/1.0.3/)