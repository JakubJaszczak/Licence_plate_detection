import json
from collections import defaultdict

import cv2
import cv2 as cv
import paddle
import torch  # noqa: F401
from paddleocr import PaddleOCR

from ocr.util import get_cropped_ground_truth_images


def main():
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", ocr_version="PP-OCRv5")
    cropped_images = get_cropped_ground_truth_images()
    output = defaultdict(dict)

    metrics = {
        "per_frame_latency_s": {},
        "total_inference_time_s": 0.0,
        "avg_inference_time_s": 0.0,
        "fps": 0.0,
        "max_vram_mb": 0.0,
        "total_images": 0,
    }

    total_inference_time_s = 0.0
    num_images = 0

    for f in sorted(cropped_images):
        img = cv.imread(str(f), cv.IMREAD_UNCHANGED)

        if len(img.shape) < 3:
            print(f"Problematic 2D image found: {f} with shape {img.shape}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        i = int(f.stem)
        orig_filename = f.parent.parent.name

        paddle.device.synchronize()
        start_event = paddle.device.cuda.Event(enable_timing=True)
        end_event = paddle.device.cuda.Event(enable_timing=True)

        start_event.record()
        results = ocr.predict(img)
        end_event.record()
        end_event.synchronize()

        inference_time_ms = start_event.elapsed_time(end_event)
        inference_time_s = inference_time_ms / 1000.0

        file_key = f"{orig_filename}/{f.name}"
        metrics["per_frame_latency_s"][file_key] = inference_time_s

        total_inference_time_s += inference_time_s
        num_images += 1

        text = results[0]["rec_texts"]
        if text:
            output[orig_filename].update({i: text})

    # with open("labels_output.json", "w") as f_out:
    #     json.dump(output, f_out, indent=4)

    if num_images > 0:
        avg_inference_time_s = total_inference_time_s / num_images
        fps = num_images / total_inference_time_s

        metrics["total_images"] = num_images
        metrics["total_inference_time_s"] = total_inference_time_s
        metrics["avg_inference_time_s"] = avg_inference_time_s
        metrics["fps"] = fps

        print(f"Przetworzono obrazow: {num_images}")
        print(f"Laczny czas inferencji: {total_inference_time_s:.4f} s")
        print(f"Sredni czas opoznienia klatki (latency): {avg_inference_time_s:.4f} s")
        print(f"Wydajnosc w klatkach na sekunde (FPS): {fps:.2f}")

    if paddle.device.is_compiled_with_cuda():
        max_vram_bytes = paddle.device.cuda.max_memory_allocated()
        max_vram_mb = max_vram_bytes / (1024 * 1024)
        metrics["max_vram_mb"] = max_vram_mb

    with open("inference_metrics.json", "w") as f_metrics:
        json.dump(metrics, f_metrics, indent=4)


if __name__ == "__main__":
    main()
