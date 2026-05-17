import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT_DIR = Path(__file__).parent.parent
CONFIDENCE_THRESHOLD = 0.5
version = 2

if __name__ == "__main__":
    weights_path = Path(
        r"D:\praca_magisterska\project\Licence_plate_detection\runs\detect\models_training\yolo\260502\weights\best.pt"
    )
    model = YOLO(weights_path)
    model_path = model.ckpt["train_args"].get("model")
    base_model_name = re.match(r".*(?P<model_name>yolov\d+[a-z]?)", model_path).group("model_name")
    results = model.predict(
        source=PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test"),
        conf=CONFIDENCE_THRESHOLD,
        project=PROJECT_ROOT_DIR.joinpath("detection", "results", base_model_name.upper()),  # The base directory
        name=f"v_{str(version).zfill(3)}",  # Sub dir
        exist_ok=True,
    )

    for r in results:
        r_np = r.cpu().numpy()
        try:
            bboxes = r_np.boxes.xyxy
        except IndexError as e:
            print(e)
            continue
        out_dir = Path(r.save_dir) / Path(r.path).stem
        r.save(
            out_dir.joinpath("results.png").as_posix(),
        )
        r.save_txt(Path(out_dir) / f"results.txt", save_conf=True)
        for i, bbox in enumerate(bboxes):
            r.save_crop(out_dir, f"{i}.png")

    print("Finished")
