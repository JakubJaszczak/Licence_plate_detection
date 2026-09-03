import re
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
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

    batch_sizes = range(1, 386, 10)
    fp16 = False
    stream = False
    source_dir = PROJECT_ROOT_DIR.joinpath("datasets", "licence_plate", "images", "test")
    out_dir = Path(f"benchmarks/yolo/with_warmup_stream-{stream}_half-{fp16}_batch")
    out_dir.mkdir(exist_ok=True, parents=True)
    summary_data = []

    for batch_size in batch_sizes:
        print(f"\n- Testowanie dla batch size: {batch_size} -")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        try:
            results = model.predict(
                source=source_dir,
                conf=CONFIDENCE_THRESHOLD,
                project=PROJECT_ROOT_DIR.joinpath("detection", "results", base_model_name.upper()),
                name=f"v_{str(version).zfill(3)}_batch_{batch_size}",
                exist_ok=True,
                batch=batch_size,
                stream=stream,
                half=fp16,
            )

            table_data = []
            for i, r in enumerate(results):
                if i % batch_size == 0:
                    pre_time = r.speed["preprocess"]
                    inf_time = r.speed["inference"]
                    post_time = r.speed["postprocess"]
                    total_time = pre_time + inf_time + post_time

                    table_data.append(
                        {
                            "Batch Index": (i // batch_size) + 1,
                            "Pre-processing (ms)": round(pre_time, 2),
                            "Inference (ms)": round(inf_time, 2),
                            "Post-processing (ms)": round(post_time, 2),
                            "Total Time (ms)": round(total_time, 2),
                        }
                    )
            df_results = pd.DataFrame(table_data)
            out_file = out_dir.joinpath(f"yolo_benchmark_batch_{batch_size}.csv")
            df_results.to_csv(out_file, index=False)
            avg_inference_time = df_results["Inference (ms)"].mean()
            avg_total_time = df_results["Total Time (ms)"].mean()
            total_fps = 1000.0 / avg_total_time if avg_total_time > 0 else 0
            inference_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0

            if torch.cuda.is_available():
                allocated_vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                allocated_vram_mb = 0.0
                peak_vram_mb = 0.0

            print(f"System throughput (Total FPS): {total_fps:.2f}")
            print(f"Peak VRAM usage: {peak_vram_mb:.2f} MB")

            summary_data.append(
                {
                    "Batch size": batch_size,
                    "Quantize": fp16,
                    "Stream": stream,
                    "Avg Pre-processing (ms)": df_results["Pre-processing (ms)"].mean(),
                    "Avg Inference time (ms)": avg_inference_time,
                    "Avg Post-processing (ms)": df_results["Post-processing (ms)"].mean(),
                    "Avg Total time (ms)": avg_total_time,
                    "Total FPS": total_fps,
                    "Inference FPS": inference_fps,
                    "Current VRAM (MB)": allocated_vram_mb,
                    "Peak VRAM (MB)": peak_vram_mb,
                }
            )
        except RuntimeError as e:
            print(str(e))
            break

    df_summary = pd.DataFrame(summary_data)
    df_summary["Timestamp"] = pd.Timestamp.now()
    df_summary.set_index("Timestamp", inplace=True)

    summary_file = out_dir.joinpath("results_summary_batches.csv")
    add_header = not summary_file.is_file()
    df_summary.to_csv(summary_file, mode="a", index=True, header=add_header)

    print("\nZakonczono pomiary. Generowanie wykresow...")
    plt.figure(figsize=(30, 15))
    plt.subplot(1, 2, 1)
    plt.plot(df_summary["Batch size"], df_summary["Total FPS"], marker="o", linestyle="-", color="b")
    plt.plot(
        df_summary["Batch size"],
        df_summary["Inference FPS"],
        marker="s",
        linestyle="--",
        color="g",
        label="Inference FPS",
    )
    plt.title("Total FPS vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Frames Per Second (FPS)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(df_summary["Batch size"], df_summary["Peak VRAM (MB)"], marker="o", linestyle="-", color="r")
    plt.title("Peak VRAM Usage vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("VRAM (MB)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"batch_size_analysis_half_{fp16}.png"))
    plt.show()
