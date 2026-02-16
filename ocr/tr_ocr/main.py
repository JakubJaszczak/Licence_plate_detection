import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

    cropped_images = Path.cwd().parent.parent.joinpath("runs/detect/predict5/license_plate")
    output = defaultdict()
    for f in sorted(cropped_images.iterdir()):
        image = Image.open(str(f)).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if generated_text:
            output[f.stem] = generated_text

    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    main()