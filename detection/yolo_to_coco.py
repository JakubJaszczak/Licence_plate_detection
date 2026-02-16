import os
from pathlib import Path

import torch
from PIL import Image
from datasets import Dataset, load_from_disk
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments

PROJECT_ROOT_DIR = Path(__file__).parent.parent
IMAGE_DIR = PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "train")
LABEL_DIR = PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "labels", "train")
MODEL_NAME = "facebook/detr-resnet-50"
NUM_CLASSES = 1
OUTPUT_DIR = "./detr_yolo"
BATCH_SIZE = 4
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
                    class_id, xc, yc, w, h = map(float, line.split())

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
                    categories.append(1)  # single class → ID = 1

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
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            single_img_annotations.append(
                {
                    "bbox": [x, y, w, h],
                    "category_id": category,
                    "iscrowd": 0,
                    "area": w * h,
                }
            )

        ann = {
            "image_id": 0,
            "annotations": single_img_annotations,
        }
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
    try:
        dataset = load_from_disk("hf_yolo_detr_dataset")
        print("Loaded dataset from disc")
    except Exception as e:
        print(e)
        print("Failed to load dataset from disc, creating dataset from Yolo labels")
        dataset = load_yolo_dataset(IMAGE_DIR, LABEL_DIR)

    dataset = dataset.with_transform(transform)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=1e-4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=detr_collate_fn,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
