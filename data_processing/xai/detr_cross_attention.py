import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

SAVE_DIR = Path(r"D:\praca_magisterska\project\Licence_plate_detection\data_processing\xai\attention_maps_results")


def process_outputs(outputs, image_path):
    # 4. FILTERING: Find which queries actually detected an object
    # outputs.logits contains the raw scores for all 100 queries across all 91 COCO classes
    probas = outputs.logits.softmax(-1)[0, :, :-1]  # Drop the last column (the "No Object" class)

    # Keep only queries where the max probability for any class is greater than 0.7 (70%)
    keep_condition = probas.max(-1).values > 0.7
    # Get the specific index numbers (e.g., [12, 45, 88]) of the successful queries
    valid_query_indices = keep_condition.nonzero().squeeze(1)

    print(f"Found {len(valid_query_indices)} valid objects. Query indices: {valid_query_indices.tolist()}")

    # 5. Extract and Combine Attention Maps
    cross_attention_maps = outputs.cross_attentions[-1]  # Shape: [batch, heads, queries, sequence]

    # If no objects were found above the threshold, exit to avoid errors
    if len(valid_query_indices) == 0:
        print("No objects detected above the confidence threshold.")
    else:
        # We will store the valid heatmaps in a list
        valid_heatmaps = []

        for q_idx in valid_query_indices:
            # Get the attention map for this specific valid query and average across the attention heads
            attn_map = cross_attention_maps[0, :, q_idx, :].mean(dim=0)
            valid_heatmaps.append(attn_map)

        # Stack them together -> Shape: [Number of Objects, Sequence Length]
        stacked_heatmaps = torch.stack(valid_heatmaps)

        # COMBINE: Take the maximum attention value at each pixel across all valid objects.
        # We use .max() instead of .sum() so the heatmap colors don't blow up in overlapping areas.
        master_attention = stacked_heatmaps.max(dim=0).values

        # 6. Reshape using the math.ceil fix
        input_h, input_w = inputs["pixel_values"].shape[-2:]
        feat_h = math.ceil(input_h / 32)
        feat_w = math.ceil(input_w / 32)

        master_heatmap_2d = master_attention.reshape(feat_h, feat_w)

        # 7. Plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(master_heatmap_2d.numpy(), cmap="jet", alpha=0.5, extent=(0, image.width, image.height, 0))
        plt.title(f"Combined Attention Map for {len(valid_query_indices)} Detected Objects")
        plt.axis("off")
        # Extract base name (e.g., "your_image") and create the save path
        base_name = image_path.stem
        save_path = SAVE_DIR.joinpath(f"{base_name}_master_heatmap.png")

        # Save the figure and close the plot to free up memory
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    # 1. Load Model and Processor
    # Important: output_attentions=True tells the model to return its "thinking"
    processor = AutoImageProcessor.from_pretrained(
        "D:\praca_magisterska\project\Licence_plate_detection\detection\detr_yolo", use_fast=True
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "D:\praca_magisterska\project\Licence_plate_detection\detection\detr_yolo", output_attentions=True
    )
    images_path = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test")
    for image_path in tqdm(images_path.iterdir()):
        # 2. Prepare Image
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")

        # 3. Forward Pass (No gradients required!)
        with torch.no_grad():
            outputs = model(**inputs)
        process_outputs(outputs, image_path)
