import matplotlib.pyplot as plt
import numpy as np

# heartbeat = (num_of_heartbeats, seq_len,cannals)
def plot_heartbeats(heartbeats):
    # heartbeat = (num_of_heartbeats, cannals, seq_len)
    heartbeats = np.transpose(heartbeats, (0, 2, 1))
    n_beats = len(heartbeats)

    # 1. Kompaktowa siatka: 3 kolumny są zazwyczaj idealne dla ekranu (np. przy 9 uderzeniach to równe 3x3)
    cols = 3
    rows = int(np.ceil(n_beats / cols))

    # 2. Mniejszy figsize, żeby wykresy były "ściśnięte" (szerokość 15, wysokość dynamiczna, ale mniejsza)
    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 2.2), sharex=True, sharey=True)

    # Zabezpieczenie, gdyby było tylko 1 uderzenie (axs nie jest wtedy tablicą)
    if n_beats == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    # 3. Definiujemy spójną paletę kolorów, żeby kanały się ładnie odcinały
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(n_beats):
        ax = axs[i]
        beat = heartbeats[i]  # Kształt: (12, próbki)

        # Rysujemy każdy z 12 kanałów (lekko cieńsze i przezroczyste linie, żeby się nie zlewały)
        for ch_idx in range(12):
            ax.plot(beat[ch_idx, :], color=colors[ch_idx], linewidth=0.8, alpha=1)

        # Tytuł mniejszy i "przyklejony" bliżej wykresu (pad=3)
        ax.set_title(f"Heartbeat #{i + 1}", fontsize=10, fontweight='bold', pad=3)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Usuwamy puste osie, jeśli siatka nie jest pełna
    for j in range(n_beats, len(axs)):
        fig.delaxes(axs[j])

    # 4. Magia czystości: JEDNA wspólna legenda na samym dole pod wszystkimi wykresami
    handles = [plt.Line2D([0], [0], color=colors[c], lw=2) for c in range(12)]
    fig.legend(handles, lead_names, loc='lower center', ncol=12, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    # 5. Ściskamy marginesy (zostawiamy tylko miejsce na dole na legendę)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.1, hspace=0.3)

    plt.show()