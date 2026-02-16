from pathlib import Path

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch

if __name__ == '__main__':
    PROJECT_ROOT_DIR = Path(__file__).parent.parent

    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    for image_path in PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate","images", "test").iterdir():
        image = Image.open(image_path)
        # prepare image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.75)

        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]
                print(f"{model.config.id2label[label]}: {score:.2f} {box}")