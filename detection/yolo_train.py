from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    model = YOLO("D:\praca_magisterska\project\Licence_plate_detection\yolov8s.pt")

    results = model.train(
        data=PROJECT_ROOT_DIR.joinpath("licence_plate.yaml"),
        epochs=100,
        batch=32,
        name=f"models_training/yolo/{datetime.now().strftime('%y%m%d')}",
        patience=5,
        device="cuda",
    )
