import json
from enum import Enum
from pathlib import Path

from ocr.ground_truth import ground_truth

current_idx = 0


class OcrResultType(Enum):
    PADDLE_LABELS = "paddle_labels"
    PADDLE_YOLO = "paddle_yolo"
    TR_LABELS = "tr_labels"
    TR_YOLO = "tr_yolo"


def create_empty_errors_json():
    errors = {}
    for key in ground_truth:
        errors[key] = {}
        for result_type in OcrResultType:
            errors[key][result_type.value] = 0

    with Path("ocr_errors.json").open("w") as f:
        json.dump(errors, f, indent=4, sort_keys=True)


def visualize():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import json
    import os
    from PIL import Image

    gt_data = ground_truth
    with open(r"D:\praca_magisterska\project\Licence_plate_detection\ocr\paddle\labels_output.json", "r") as f:
        paddle_labels = json.load(f)
    with open(r"D:\praca_magisterska\project\Licence_plate_detection\ocr\paddle\yolo_output.json", "r") as f:
        paddle_yolo = json.load(f)
    with open(r"D:\praca_magisterska\project\Licence_plate_detection\ocr\tr_ocr\labels_output.json", "r") as f:
        tr_labels = json.load(f)
    with open(r"D:\praca_magisterska\project\Licence_plate_detection\ocr\tr_ocr\yolo_output.json", "r") as f:
        tr_yolo = json.load(f)

    # 2. Extract keys ONLY from the ground truth file
    # This guarantees we only iterate through your 100 target images
    all_keys = sorted(list(gt_data.keys()))
    image_folder = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test")

    # 3. Set up a 2x3 Grid Layout
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[3, 1, 1, 1])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_p1 = fig.add_subplot(gs[0, 2])
    ax_p2 = fig.add_subplot(gs[0, 3])
    ax_p3 = fig.add_subplot(gs[1, 1])
    ax_p4 = fig.add_subplot(gs[1, 2])

    axes = [ax_img, ax_gt, ax_p1, ax_p2, ax_p3, ax_p4]

    def draw_current_view():
        # Clear all subplots
        for ax in axes:
            ax.clear()
            ax.axis("off")

        if not all_keys:
            return

        key = all_keys[current_idx]
        fig.suptitle(f"Image Key: {key} ({current_idx + 1}/{len(all_keys)})", fontsize=16)

        # Draw Image
        img_path = image_folder.joinpath(f"{key}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax_img.imshow(img)
            ax_img.set_title("Original Image")
        else:
            ax_img.text(0.5, 0.5, "Image Not Found", ha="center", va="center")

        def draw_text(ax, title, data_dict):
            text_val = json.dumps(data_dict.get(key, {"error": "No prediction"}), indent=4)
            ax.set_title(title)
            ax.text(
                0.05,
                0.95,
                text_val,
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=10,
                bbox=dict(facecolor="whitesmoke", alpha=0.8, edgecolor="none"),
            )

        # Draw all JSON texts
        draw_text(ax_gt, "Ground Truth", gt_data)
        draw_text(ax_p1, "Paddle labels", paddle_labels)
        draw_text(ax_p2, "Paddle Yolo", paddle_yolo)
        draw_text(ax_p3, "TR labels", tr_labels)
        draw_text(ax_p4, "TR Yolo", tr_yolo)

        fig.canvas.draw()

    axprev = fig.add_axes([0.7, 0.02, 0.1, 0.05])
    axnext = fig.add_axes([0.81, 0.02, 0.1, 0.05])
    bprev = Button(axprev, "Previous")
    bnext = Button(axnext, "Next")

    def next_image(event):
        global current_idx
        current_idx = (current_idx + 1) % len(all_keys)
        draw_current_view()

    def prev_image(event):
        global current_idx
        current_idx = (current_idx - 1) % len(all_keys)
        draw_current_view()

    bnext.on_clicked(next_image)
    bprev.on_clicked(prev_image)

    draw_current_view()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


if __name__ == "__main__":
    # root = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test")
    # for img_path in root.iterdir():
    #     if img_path.suffix != ".jpg":
    #         continue
    #     open_and_wait(img_path)

    visualize()
