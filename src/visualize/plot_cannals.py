import matplotlib.pyplot as plt

# this func is craeted for plotting all 12 cannals of ECG seq
# signal = (cannals, seq_len)
def plot_full_ecg(signal, title="Full 10 sec, 12 channels ECG sequence"):
    """
        Visualizes a standard 12-lead ECG signal in a compact, clinical-style layout.

        The function arranges 12 leads into a 6x2 grid, sharing axes for easy
        morphological comparison. It uses internal text labels for channel names
        to maximize plot area and removes redundant frame spines for a clean look.

        Args:
            signal (np.ndarray): The input ECG data. Expects shape (12, seq_len)
                or (seq_len, 12).
            title (str, optional): Main title of the figure. Defaults to
                "Full 10 sec, 12 channels ECG sequence".

        Returns:
            None: Displays the generated plot using matplotlib.pyplot.show().

        Note:
            - The function automatically transposes the input if the shape is (seq_len, 12).
            - Shared axes (sharex=True, sharey=True) are used to allow precise
              timing and amplitude comparison across all leads.
        """

    # Upewniamy się, że sygnał ma kształt (12, próbki)
    if signal.shape[0] != 12:
        signal = signal.T

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Rozmiar 15x8 jest idealny na standardowe ekrany (kompaktowy, ale czytelny)
    fig, axs = plt.subplots(6, 2, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)

    axs = axs.flatten()

    for i in range(12):
        ax = axs[i]

        # Rysujemy sygnał (niebieski, czytelny kolor, bez grubych linii)
        ax.plot(signal[i, :], color='#1f77b4', linewidth=1.0)

        # MAGIA: Zamiast ax.set_title() (które marnuje miejsce),
        # wrzucamy nazwę kanału do środka wykresu w lewy górny róg.
        ax.text(0.01, 0.80, lead_names[i], transform=ax.transAxes,
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        # Cienka siatka w tle dla ułatwienia oceny morfologii
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

        # Pozbywamy się ciężkich, górnych i prawych ramek dla czystości
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Mniej cyferek na osi Y w prawej kolumnie, żeby się nie zlewały
        if i % 2 != 0:
            ax.tick_params(labelleft=False)

    # Ekstremalne zacieśnienie marginesów
    # hspace=0.05 sprawia, że wykresy w pionie są niemal przyklejone do siebie
    plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.05, wspace=0.05, hspace=0.05)

    plt.show()

# Wywołanie:
# plot_full_ecg(X[0])