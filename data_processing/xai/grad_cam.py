import glob
import os
from pathlib import Path

import cv2
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.ops import box_iou
from ultralytics import YOLO

# --- CONFIGURATION ---
IMG_DIR = r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test"
SAVE_DIR = "grad_cam_results/"
MODEL_PATH = (
    r"D:\praca_magisterska\project\Licence_plate_detection\runs\detect\models_training\yolo\260502\weights\best.pt"
)
MODEL_INPUT_SIZE = (640, 640)
LABELS_ROOT = Path(r"D:\praca_magisterska\project\Licence_plate_detection\detection\results\YOLOV8S\v_002")
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLOWrapper(torch.nn.Module):
    """Wraps the Ultralytics model to return only the raw prediction tensor."""

    def __init__(self, pt_model_path):
        super().__init__()

        # 1. Ładujemy model YOLO do zmiennej LOKALNEJ (bez 'self.')
        # Dzięki temu obiekt YOLO nie staje się submodułem PyTorcha
        temp_yolo = YOLO(pt_model_path)

        # 2. Wyciągamy sam czysty model PyTorch (warstwy konwolucyjne itp.)
        # i przypisujemy go do self.model
        self.model = temp_yolo.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Forward pass używa teraz wyłącznie czystego modelu PyTorch
        preds = self.model(x)
        if isinstance(preds, (tuple, list)):
            return preds[0]
        return preds


class YOLOBoxTarget:
    """Tells Grad-CAM which specific anchor box's gradients to backpropagate."""

    def __init__(self, target_box_xyxy, class_idx=0):
        # target_box_xyxy must be scaled to the input tensor size (e.g., 0-640)
        self.target_box = torch.tensor([target_box_xyxy], dtype=torch.float32).to(device)
        self.class_idx = class_idx

    def __call__(self, model_outputs):
        # 1. Transpose output to easily loop through anchors: [1, 5, 8400] -> [8400, 5]
        preds = model_outputs.squeeze(0).transpose(0, 1)

        # 2. Extract predicted center coords, width, and height
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]

        # 3. Convert YOLO predicted centers to corner coordinates [xmin, ymin, xmax, ymax]
        pred_boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        # 4. Find which of the 8400 anchors matches our .txt file box best
        # Calculate Intersection over Union (IoU)
        ious = box_iou(pred_boxes, self.target_box)  # Shape: [8400, 1]
        best_anchor_idx = ious.argmax()

        # 5. Return the class score of that specific anchor
        # Gradients will now ONLY flow backward from this exact number
        score = preds[best_anchor_idx, 4 + self.class_idx]

        print(f"  -> Best Anchor IoU: {ious.max().item():.4f} | Anchor Confidence: {score.item():.4f}")
        return score


if __name__ == "__main__":
    print("Loading PyTorch model...")
    model_wrapper = YOLOWrapper(MODEL_PATH).to(device)

    # Target all three scales (Small, Medium, Large)
    m = model_wrapper.model.model
    target_layers = [m[15], m[18], m[21]]

    # Use GradCAMPlusPlus: It is mathematically designed to handle multiple target
    # layers and multiple objects much better than the base GradCAM.
    from pytorch_grad_cam import GradCAMPlusPlus

    cam = GradCAMPlusPlus(model=model_wrapper, target_layers=target_layers)

    # ==========================================
    # 3. BATCH PROCESSING
    # ==========================================

    image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    print(f"Found {len(image_paths)} images.")

    for img_path in image_paths:
        image_name = Path(img_path).stem
        label_path = LABELS_ROOT.joinpath(image_name, "results.txt")
        if not os.path.exists(label_path):
            continue

        # Load image using OpenCV for standard processing
        raw_img = cv2.imread(img_path)
        orig_h, orig_w = raw_img.shape[:2]

        # Resize to 640x640 and normalize to [0, 1] for Grad-CAM overlay
        img_resized = cv2.resize(raw_img, MODEL_INPUT_SIZE)
        rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0

        # Create PyTorch tensor [Batch, Channels, Height, Width]
        input_tensor = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Parse the YOLO .txt file
        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 6:
                continue
            cls, xc, yc, w, h, conf = [float(x) for x in parts]

            # 1. THE LOGICAL FILTER
            # If the model had less than 20% confidence, the gradient will be almost 0.
            # Skip generating an empty map for boxes the model essentially ignored.
            if conf < 0.20:
                print(f"Skipping Object {i} - Confidence too low ({conf:.4f})")
                continue

            # Calculate coordinates normalized to the 640x640 tensor
            xmin = (xc - w / 2) * MODEL_INPUT_SIZE[0]
            ymin = (yc - h / 2) * MODEL_INPUT_SIZE[1]
            xmax = (xc + w / 2) * MODEL_INPUT_SIZE[0]
            ymax = (yc + h / 2) * MODEL_INPUT_SIZE[1]

            target_box_xyxy = [xmin, ymin, xmax, ymax]

            # 1. Define the specific target
            targets = [YOLOBoxTarget(target_box_xyxy, class_idx=0)]

            # 2. Generate the heatmap (returns a 2D numpy array [640, 640])
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)[
                0, :
            ]  # 3. DIAGNOSTIC PRINT: Check if the heatmap is truly empty
            cam_max = grayscale_cam.max()
            print(f"    -> CAM Strength for Object {i}: {cam_max:.6f}")

            if cam_max < 1e-7:
                print("       [Warning] Heatmap is mathematically empty. Check layer indices.")

            # 3. Create the visual overlay
            if grayscale_cam.max() > 0:
                grayscale_cam = grayscale_cam / grayscale_cam.max()

                # 4. VISUALIZATION
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # 4. Draw the bounding box for clarity
            cv2.rectangle(visualization, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)

            # Convert back to BGR for OpenCV saving and resize back to original image size
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            final_output = cv2.resize(visualization_bgr, (orig_w, orig_h))

            # Save
            base_name = os.path.basename(img_path).replace(".jpg", "")
            save_path = os.path.join(SAVE_DIR, f"{base_name}_obj{i}.jpg")
            cv2.imwrite(save_path, final_output)

    print(f"Finished! Native PyTorch heatmaps saved to '{SAVE_DIR}'.")
