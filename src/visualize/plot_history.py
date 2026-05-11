import matplotlib.pyplot as plt


def plot_training_history(train_history, val_history, model_title = "", path=None):
    """
    Creates a flexible visualization of training and validation history.
    Automatically detects metrics present in the history dictionaries.

    Args:
        train_history (list): List of dictionaries with training metrics.
        val_history (list): List of dictionaries with validation metrics.
    """
    if not train_history:
        print("No training history provided.")
        return

    # Extract common variables
    steps = [d['step'] for d in train_history]

    # Identify metrics dynamically (exclude step and lr)
    all_keys = train_history[0].keys()
    metrics = [k for k in all_keys if k not in ['step', 'lr']]

    # Setup plot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Training metrics plot (plots all found keys dynamically)
    for metric in metrics:
        values = [d[metric] for d in train_history]
        ax1.plot(steps, values, label=metric.replace('_', ' ').capitalize(), linewidth=2)

    ax1.set_title(f'Training History for {model_title}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Validation metrics plot
    if val_history:
        val_steps = [d['step'] for d in val_history]
        val_keys = [k for k in val_history[0].keys() if k != 'step']

        for metric in val_keys:
            values = [d[metric] for d in val_history]
            ax2.plot(val_steps, values, label=metric.replace('_', ' ').capitalize(),
                     marker='s', linewidth=2)

        ax2.set_title(f'Validation History for {model_title}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Validation Data', ha='center', va='center')

    plt.tight_layout()

    if path is not None:
        print("[INFO] Saving history to", path)
        plt.savefig(path)

    if path is None:
        plt.show()

    plt.close()
