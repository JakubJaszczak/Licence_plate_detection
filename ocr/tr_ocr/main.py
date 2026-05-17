import json
from collections import defaultdict

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr.util import get_cropped_ground_truth_images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

    # cropped_images = get_cropped_images(DetectionModelType.YOLO, "v_002")
    cropped_images = get_cropped_ground_truth_images()
    output = defaultdict(dict)
    for f in sorted(cropped_images):
        image = Image.open(str(f)).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        i = int(f.stem)
        orig_filename = f.parent.parent.name
        if generated_text:
            output[orig_filename].update({i: generated_text})

    with open("labels_output.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
