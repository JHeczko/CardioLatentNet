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
    
## Deliverables
- notebooki 
- raport PDF
- wykresy latent‑space.

## Links
- [NATURE](https://www.nature.com/articles/s41597-020-0495-6)
- [PHYSIONET](https://physionet.org/content/ptb-xl/1.0.3/)