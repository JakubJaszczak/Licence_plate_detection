import argparse
import math
import os
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from datasets import Dataset, load_from_disk
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments, EarlyStoppingCallback

PROJECT_ROOT_DIR = Path(__file__).parent.parent
IMAGE_DIR = PROJECT_ROOT_DIR.joinpath(
    "datasets",
    "licence_plate",
    "images",
)
LABEL_DIR = PROJECT_ROOT_DIR.joinpath(
    "datasets",
    "licence_plate",
    "labels",
)
MODEL_NAME = "facebook/detr-resnet-50"
NUM_CLASSES = 1
OUTPUT_DIR = "./detr_v3"
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 150
LR = 1e-4


def load_yolo_dataset(image_root, label_root, mode: Literal["train", "val"]):
    data = []
    image_dir = image_root.joinpath(mode)
    label_dir = label_root.joinpath(mode)

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

                    boxes.append([x_min, y_min, box_w, box_h])
                    categories.append(0)  # single class -> ID = 0

        data.append({"image_path": img_path, "objects": {"bbox": boxes, "category": categories}})

    ds = Dataset.from_list(data)
    ds.save_to_disk(f"{mode}_dataset")
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
    parser.add_argument(
        "--fp16", action="store_true", help="Enable mixed precision (can be unstable for DETR on small GPUs)."
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        # Helps reduce fragmentation-related OOMs on long runs.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    try:
        if args.rebuild_dataset:
            raise FileNotFoundError("Forced dataset rebuild")
        dataset = load_from_disk("train_dataset")
        val_dataset = load_from_disk("val_dataset")
        print("Loaded dataset from disk")
    except Exception as e:
        print(e)
        print("Failed to load dataset from disk, creating dataset from YOLO labels")
        dataset = load_yolo_dataset(IMAGE_DIR, LABEL_DIR, "train")
        val_dataset = load_yolo_dataset(IMAGE_DIR, LABEL_DIR, "val")

    train_dataset = dataset.with_transform(transform)
    val_dataset = val_dataset.with_transform(transform)

    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    for param in model.model.backbone.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        weight_decay=1e-4,
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        fp16=False,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=detr_collate_fn,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
