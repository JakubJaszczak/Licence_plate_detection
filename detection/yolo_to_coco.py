import os
from pathlib import Path
import argparse
import math

import torch
from PIL import Image
from datasets import Dataset, load_from_disk
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments

PROJECT_ROOT_DIR = Path(__file__).parent.parent
print(PROJECT_ROOT_DIR)
IMAGE_DIR = Path("/home/karol/Downloads/large-license-plate-dataset/images/train")
LABEL_DIR = Path("/home/karol/Downloads/large-license-plate-dataset/labels/train")
MODEL_NAME = "facebook/detr-resnet-50"
NUM_CLASSES = 1
OUTPUT_DIR = "./detr_yolo"
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 10
LR = 1e-4


def load_yolo_dataset(image_dir, label_dir):
    data = []

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.rsplit(".", 1)[0] + ".txt")

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        boxes = []
        categories = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id, xc, yc, w, h = map(float, parts)
                    if not all(math.isfinite(v) for v in (xc, yc, w, h)):
                        continue
                    if w <= 0 or h <= 0:
                        continue

                    x_min = (xc - w / 2) * W
                    y_min = (yc - h / 2) * H
                    box_w = w * W
                    box_h = h * H

                    # import numpy as np
                    # import cv2 as cv
                    # np_img = np.array(image)
                    # # RGB → BGR (what OpenCV expects)
                    # cv_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
                    # img_with_bbox = cv.rectangle(
                    #     cv_img,
                    #     (int(x_min), int(y_min)),
                    #     (int(x_min + box_w), int(y_min + box_h)),
                    #     (0, 0, 255),  # red (BGR)
                    #     2
                    # )

                    boxes.append([x_min, y_min, box_w, box_h])
                    categories.append(0)  # single class -> ID = 0

        data.append({"image_path": img_path, "objects": {"bbox": boxes, "category": categories}})

    ds = Dataset.from_list(data)
    ds.save_to_disk("hf_yolo_detr_dataset")
    return ds


processor = DetrImageProcessor.from_pretrained(MODEL_NAME)


def transform(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]

    annotations = []

    for i, obj in enumerate(examples["objects"]):
        single_img_annotations = []
        for bbox, category in zip(obj["bbox"], obj["category"]):
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            img_width, img_height = images[i].size
            x = min(max(0.0, float(x)), float(img_width))
            y = min(max(0.0, float(y)), float(img_height))
            w = min(float(w), float(img_width) - x)
            h = min(float(h), float(img_height) - y)
            if w <= 1e-6 or h <= 1e-6:
                continue

            single_img_annotations.append(
                {
                    "bbox": [x, y, w, h],
                    "category_id": category,
                    "iscrowd": 0,
                    "area": w * h,
                }
            )

        ann = {"image_id": i, "annotations": single_img_annotations}
        annotations.append(ann)

    encoding = processor(images=images, annotations=annotations, return_tensors="pt")

    return encoding


def detr_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]  # list of dicts

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR on YOLO-format license plate data.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild hf_yolo_detr_dataset from YOLO labels.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (can be unstable for DETR on small GPUs).")
    args = parser.parse_args()

    if torch.cuda.is_available():
        # Helps reduce fragmentation-related OOMs on long runs.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    try:
        if args.rebuild_dataset:
            raise FileNotFoundError("Forced dataset rebuild")
        dataset = load_from_disk("hf_yolo_detr_dataset")
        print("Loaded dataset from disk")
    except Exception as e:
        print(e)
        print("Failed to load dataset from disk, creating dataset from YOLO labels")
        dataset = load_yolo_dataset(IMAGE_DIR, LABEL_DIR)

    dataset = dataset.with_transform(transform)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        fp16=args.fp16 and torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=detr_collate_fn,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
####  odpal to python detection/yolo_to_coco.py --rebuild-dataset --batch-size 1 --grad-accum-steps 2 --fp16
