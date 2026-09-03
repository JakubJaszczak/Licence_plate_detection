import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    json_path = Path(
        r"D:\praca_magisterska\project\Licence_plate_detection\detection\detr_high_patience\checkpoint-732320\trainer_state.json"
    )
    with json_path.open() as f:
        logs = json.load(f)["log_history"]
    mode = "training"
    if mode == "training":
        training_logs = list(filter(lambda l: l["epoch"] % 1 != 0, logs))
        loss = [l["loss"] for l in training_logs]
        epoch = [l["epoch"] for l in training_logs]
    elif mode == "validation":
        val_logs = list(filter(lambda x: "eval_loss" in x, logs))
        loss = [l["eval_loss"] for l in val_logs]
        epoch = [l["epoch"] for l in val_logs]
    else:
        loss = []
        epoch = []

    plt.figure(figsize=(20, 12))
    plt.plot(epoch, loss, linestyle="-", color="#1f77b4", linewidth=1)
    plt.title(
        f"Funkcja straty na zbiorze {'treningowym' if mode == 'training' else 'walidacyjnym'} ({mode.capitalize()} loss) w zależnosci od epoki",
        fontsize=14,
    )
    plt.xlabel("Epoka", fontsize=12)
    plt.ylabel("Wartość funkcji straty", fontsize=12)

    z = np.polyfit(epoch, loss, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(epoch), max(epoch), 100)
    y_smooth = p(x_smooth)
    plt.plot(x_smooth, y_smooth, linestyle="-", color="greenyellow", linewidth=2.5, label="Dopasowana linia trendu")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    out_path = Path("detr_training_plots").joinpath(json_path.parent.parent.stem + "_" + mode).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.show()
