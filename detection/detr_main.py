import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_NAME = "DETR"
version = 2
# v_str = f"v_{str(version).zfill(3)}"
v_str = "my_dataset_detr_v1"
label_name = "licence_plate"


def bbox_to_normalized(bbox, img_size):
    h, w = img_size
    x1, y1, x2, y2 = bbox
    return x1 / w, y1 / h, x2 / w, y2 / h


def get_random_color(seed):
    color_int = random.Random(seed).randint(0, 256 * 256 * 256 - 1)
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return r, g, b


if __name__ == "__main__":
    PROJECT_ROOT_DIR = Path(__file__).parent.parent

    image_processor = AutoImageProcessor.from_pretrained("./detr_yolo", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained("./detr_yolo")
    row = "{sem_class} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}"
    row_px = "{sem_class} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.6f}"
    # for image_path in tqdm(PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test").iterdir()):
    for image_path in tqdm(
        Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\my_dataset\test").iterdir()
    ):
        image = Image.open(image_path)
        img_np = np.array(image)

        # prepare image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        results = image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.75
        )
        bboxes_img = img_np.copy()
        image_root_dir = PROJECT_ROOT_DIR / "detection" / "results" / MODEL_NAME / v_str / image_path.stem
        image_root_dir.mkdir(exist_ok=True, parents=True)
        txt_output = []
        txt_output_pixels = []
        for result in results:
            assert len(results) == 1
            for i, (score, label_id, bbox) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
                tmp_img = img_np.copy()
                score, label = score.item(), label_id.item()
                bbox_int = [int(i) for i in bbox.tolist()]
                x1_norm, y1_norm, x2_norm, y2_norm = bbox_to_normalized(bbox.numpy(), tmp_img.shape[:2])
                txt_output.append(
                    row.format(sem_class=label, x1=x1_norm, y1=y1_norm, x2=x2_norm, y2=y2_norm, conf=score)
                )

                x1, y1, x2, y2 = bbox_int
                txt_output_pixels.append(row_px.format(sem_class=label, x1=x1, y1=y1, x2=x2, y2=y2, conf=score))
                h, w = tmp_img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                cropped_img = tmp_img[y1:y2, x1:x2]
                c = get_random_color(str((i, image_path.name)))
                cv2.rectangle(bboxes_img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(bboxes_img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)
                filename = image_root_dir / label_name / f"{i}.png"
                filename.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(cropped_img).save(filename)

        Image.fromarray(bboxes_img).save(image_root_dir / "results.png")
        with image_root_dir.joinpath("results.txt").open("w") as f:
            f.write("\n".join(txt_output))
