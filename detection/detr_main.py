import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection


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
    for image_path in tqdm(PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test").iterdir()):
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

        for result in results:
            assert len(results) == 1
            for i, (score, label_id, bbox) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
                tmp_img = img_np.copy()
                score, label = score.item(), label_id.item()
                bbox = [int(i) for i in bbox.tolist()]
                x1, y1, x2, y2 = bbox
                h, w = tmp_img.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                cropped_img = tmp_img[y1:y2, x1:x2]
                c = get_random_color(str((i, image_path.name)))
                cv2.rectangle(bboxes_img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(bboxes_img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)
                filename = (PROJECT_ROOT_DIR / "detection/cropped_images" / f"{i}_{image_path.stem}").with_suffix(
                    ".png"
                )
                Image.fromarray(cropped_img).save(filename)
                # print(f"{model.config.id2label[label]}: {score:.2f} {bbox}")

        Image.fromarray(bboxes_img).save(
            (PROJECT_ROOT_DIR / "detection/bboxes_images" / image_path.stem).with_suffix(".png")
        )
