from enum import Enum, auto
from pathlib import Path

from ultralytics import YOLO

class ModelVersion(Enum):
    YOLO11n = auto()
    YOLOv8n = auto()
    YOLOv8s = auto()


# TODO take from yaml
PROJECT_ROOT_DIR = Path(__file__).parent.parent
CONFIDENCE_THRESHOLD = 0.7
if __name__ == '__main__':
    weights_path = PROJECT_ROOT_DIR.joinpath("runs/detect/train4/weights/best.pt")
    model = YOLO(weights_path)
    # for img in PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate","images", "test").iterdir():
    #     results = model.predict(PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate","images", "test"))
    #     break

    results = model.predict(PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test"), conf=CONFIDENCE_THRESHOLD)

    for r in results:
        print(r.boxes.data)
        r_np = r.cpu().numpy()
        # First bbox is the one with the highest conf, for now only try the first one
        try:
            bbox = r_np.boxes.xyxy[0]
        except IndexError as e:
            print(e)
            continue
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_img = r.orig_img[y1:y2, x1:x2]
        filename = Path(r.path).name
        r.save_crop(r.save_dir, filename)
    print("Finished")