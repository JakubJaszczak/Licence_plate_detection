import datetime
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_NAME = "DETR"
version = 4
v_str = f"{datetime.datetime.now().strftime('%Y%m%d')}_v_{str(version).zfill(3)}"
# v_str = "my_dataset_detr_v1"
label_name = "licence_plate"
THRESHOLD = 0.5


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


row = "{sem_class} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}"
row_px = "{sem_class} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.6f}"
if __name__ == "__main__":
    PROJECT_ROOT_DIR = Path(__file__).parent.parent

    image_processor = AutoImageProcessor.from_pretrained("./detr_v2", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained("./detr_v2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_dataset_dir = rf"{PROJECT_ROOT_DIR}\datasets\licence_plate\images\test"
    image_paths = list(Path(test_dataset_dir).iterdir())
    total_images = len(image_paths)

    BATCH_SIZE = 12

    # --- WARM UP ---
    print(f"Warm up (rozmiar paczki: {BATCH_SIZE})...")
    dummy_pixel_values = torch.randn(BATCH_SIZE, 3, 800, 800).to(device)
    dummy_pixel_mask = torch.ones(BATCH_SIZE, 800, 800, dtype=torch.int64).to(device)
    dummy_inputs = {"pixel_values": dummy_pixel_values, "pixel_mask": dummy_pixel_mask}
    for _ in range(20):
        with torch.no_grad():
            _ = model(**dummy_inputs)

    del dummy_pixel_values, dummy_pixel_mask, dummy_inputs
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    batch_times = []
    batch_fps_list = []

    print(f"Rozpoczynanie właściwej inferencji dla {total_images} obrazów...")

    for i in tqdm(range(0, total_images, BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        current_batch_size = len(batch_paths)
        images = []
        target_sizes = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
            target_sizes.append(img.size[::-1])
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            outputs = model(**inputs)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = torch.cuda.Event.elapsed_time(start_event, end_event)
        inference_time_seconds = elapsed_time_ms / 1000.0

        fps = current_batch_size / inference_time_seconds

        batch_times.append(inference_time_seconds)
        batch_fps_list.append(fps)

        print(f"Paczka {len(batch_times)}: czas = {inference_time_seconds:.4f} s, FPS = {fps:.2f}")

        del inputs, outputs, images
        if device == "cuda":
            torch.cuda.empty_cache()

    output_dir = Path("benchmarks/detr")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = f"wyniki_inferencji_batch_size_{BATCH_SIZE}.txt"
    allocated_vram_mb = torch.cuda.memory_allocated(device) / (1024**2)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    with open(output_dir.joinpath(output_file), "w", encoding="utf-8") as f:
        f.write("=== RAPORT Z INFERENCJI MODELU DETR ===\n")
        f.write(f"Całkowita liczba obrazów: {total_images}\n")
        f.write(f"Zastosowany rozmiar paczki (BATCH_SIZE): {BATCH_SIZE}\n\n")
        f.write("Szczegóły dla poszczególnych paczek:\n")
        f.write("-" * 50 + "\n")

        for idx, (b_time, b_fps) in enumerate(zip(batch_times, batch_fps_list)):
            f.write(f"Paczka {idx + 1}: Czas = {b_time:.4f} s | Wydajność = {b_fps:.2f} FPS\n")

        f.write("-" * 50 + "\n")
        avg_time = sum(batch_times) / len(batch_times)
        avg_fps = sum(batch_fps_list) / len(batch_fps_list)
        f.write(f"Średni czas inferencji paczki: {avg_time:.4f} s\n")
        f.write(f"Średni wskaźnik wydajności: {avg_fps:.2f} FPS\n")
        f.write(f"Maksymalne zużycie VRAM: {peak_vram_mb} MB\n")
        f.write(f"Aktualne zużycie VRAM: {allocated_vram_mb} MB\n")

    print(f"\nWyniki zostały pomyślnie zapisane do pliku: {output_file}")
