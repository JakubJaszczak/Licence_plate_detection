import json
import math
from collections import defaultdict

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr.util import get_cropped_ground_truth_images


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

    cropped_images = get_cropped_ground_truth_images()
    output = defaultdict(dict)

    images_list = []
    paths_list = []

    for f in sorted(cropped_images):
        image = Image.open(str(f)).convert("RGB")
        images_list.append(image)
        paths_list.append(f)

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_inference_time_ms = 0.0
    fps = 0.0

    if images_list:
        mid_idx = math.ceil(len(images_list) / 2)
        batches = [(images_list[:mid_idx], paths_list[:mid_idx]), (images_list[mid_idx:], paths_list[mid_idx:])]

        for batch_images, batch_paths in batches:
            if not batch_images:
                continue

            pixel_values = processor(batch_images, return_tensors="pt").pixel_values.to(device)

            start_event.record()
            generated_ids = model.generate(pixel_values)
            end_event.record()
            torch.cuda.synchronize()
            total_inference_time_ms += start_event.elapsed_time(end_event)

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for f, generated_text in zip(batch_paths, generated_texts):
                i = int(f.stem)
                orig_filename = f.parent.parent.name
                if generated_text:
                    output[orig_filename].update({i: generated_text})

        total_inference_time_s = total_inference_time_ms / 1000
        if total_inference_time_s > 0:
            fps = len(images_list) / total_inference_time_s

    print(f"Calkowity czas inferencji dla 2 batchy: {total_inference_time_ms / 1000:.4f} s")
    print(f"FPS inferencji: {fps:.2f} obrazow/s")
    peak_vram = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram / (1024 * 1024)
    print(f"Szczytowe zuzycie pamieci VRAM: {peak_vram_mb:.2f} MB")

    # with open("labels_output.json", "w") as f:
    #     json.dump(output, f, indent=4)

    metrics = {
        "num_images": len(images_list),
        "total_inference_time_s": total_inference_time_ms / 1000,
        "fps": fps,
        "peak_vram_mb": peak_vram_mb,
    }

    with open("inference_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
