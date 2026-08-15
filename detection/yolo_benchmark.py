import re
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT_DIR = Path(__file__).parent.parent
CONFIDENCE_THRESHOLD = 0.5
version = 2

if __name__ == "__main__":
    weights_path = Path(
        r"D:\praca_magisterska\project\Licence_plate_detection\runs\detect\models_training\yolo\260502\weights\best.pt"
    )
    device = torch.device("cuda:0")
    model = YOLO(weights_path).to(device)
    model_path = model.ckpt["train_args"].get("model")
    base_model_name = re.match(r".*(?P<model_name>yolov\d+[a-z]?)", model_path).group("model_name")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    results = model.predict(
        source=PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test"),
        conf=CONFIDENCE_THRESHOLD,
        project=PROJECT_ROOT_DIR.joinpath("detection", "results", base_model_name.upper()),  # The base directory
        name=f"v_{str(version).zfill(3)}",  # Sub dir
        exist_ok=True,
    )

    table_data = []

    for i, r in enumerate(results):
        pre_time = r.speed["preprocess"]
        inf_time = r.speed["inference"]
        post_time = r.speed["postprocess"]
        total_time = pre_time + inf_time + post_time

        table_data.append(
            {
                "Frame": i + 1,
                "Pre-processing (ms)": round(pre_time, 2),
                "Inference (ms)": round(inf_time, 2),
                "Post-processing (ms)": round(post_time, 2),
                "Total Time (ms)": round(total_time, 2),
            }
        )

    df_results = pd.DataFrame(table_data)
    print("Table with times for individual frames:")
    print(df_results.to_string(index=False))

    avg_inference_time = df_results["Inference (ms)"].mean()
    avg_total_time = df_results["Total Time (ms)"].mean()

    total_fps = 1000.0 / avg_total_time
    inference_fps = 1000.0 / avg_inference_time
    if torch.cuda.is_available():
        allocated_vram_mb = torch.cuda.memory_allocated(device) / (1024**2)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        allocated_vram_mb = 0.0
        peak_vram_mb = 0.0

    # Display statistical summary
    print("\n- Statistical Summary -")
    print(f"Average pre-processing time: {df_results['Pre-processing (ms)'].mean():.2f} ms")
    print(f"Average inference time (pure): {avg_inference_time:.2f} ms")
    print(f"Average post-processing time: {df_results['Post-processing (ms)'].mean():.2f} ms")
    print(f"Average total time per frame: {avg_total_time:.2f} ms")
    print(f"System throughput (Total FPS): {total_fps:.2f}")
    print(f"Theoretical FPS (Inference only): {inference_fps:.2f}")
    print(f"Current VRAM usage: {allocated_vram_mb:.2f} MB")
    print(f"Peak VRAM usage: {peak_vram_mb:.2f} MB")
