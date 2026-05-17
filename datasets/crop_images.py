from pathlib import Path

import numpy as np
from PIL import Image

root = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test")
labels_path = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\labels\test")
out_dir = Path("D:\praca_magisterska\project\Licence_plate_detection\datasets\cropped_test_images")
if __name__ == "__main__":
    for img in root.iterdir():
        img_name = img.name

        label_path = labels_path / img_name.replace(".jpg", ".txt")
        img_path = root / img_name
        img_np = np.array(Image.open(img_path))
        H, W = img_np.shape[:2]

        with label_path.open("r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.split()
            cls, xc, yc, w, h = map(float, parts)
            xmin = (xc - w / 2) * W
            ymin = (yc - h / 2) * H
            xmax = (xc + w / 2) * W
            ymax = (yc + h / 2) * H

            cropped_img = img_np[int(ymin) : int(ymax), int(xmin) : int(xmax)]
            out_filepath = out_dir / img.stem / "licence_plate" / f"{i}.png"
            out_filepath.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cropped_img).save(out_filepath)
