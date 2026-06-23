from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

OUT_DIR = Path("better_grad_cam")


class YOLOv8Wrapper(nn.Module):
    def __init__(self, yolov8_model):
        super().__init__()
        self.model = yolov8_model.model

    def forward(self, x):
        # Forward pass through the raw model
        res = self.model(x)
        # YOLOv8 returns a tuple during inference.
        # The first element is the raw prediction tensor of shape: [batch, 84, 8400]
        return res[0] if isinstance(res, (list, tuple)) else res


class YOLOv8Target:
    def __init__(self, target_class_idx):
        self.class_idx = target_class_idx

    def __call__(self, model_output):
        # model_output shape: [batch_size, 84, 8400] for COCO (4 bbox coords + 80 classes)
        # We slice the tensor to get the probabilities for our specific target class
        class_probs = model_output[4 + self.class_idx, :]
        # We return the maximum probability across all 8400 anchors to compute the gradient
        return class_probs.max()


layers = {
    "grad_cam": [
        4,
        6,  # 4,6 -> The Texture and Edge Layers (Mid-Backbone)
        9,  # -> The Context Layer (SPPF)
        15,  # small objects
        18,  # medium objects
        21,  # large objects
    ],
    "eigen_cam": [9, 15, 18, 21],
}
if __name__ == "__main__":
    base_model = YOLO(
        r"D:\praca_magisterska\project\Licence_plate_detection\runs\detect\models_training\yolo\260502\weights\best.pt"
    )
    wrapped_model = YOLOv8Wrapper(base_model)
    wrapped_model.eval()

    for param in wrapped_model.parameters():
        param.requires_grad = True

    images_root = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\test")
    for img_path in images_root.iterdir():
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError("Image not found. Please provide a valid path.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_float = img_resized.astype(np.float32) / 255.0
        targets = [YOLOv8Target(target_class_idx=0)]  # 0 is a class id of a licence plate in a trained dataset
        input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor.requires_grad_(True)

        print("\n--- Running Grad-CAM ---")
        for layer_idx in layers["grad_cam"]:
            target_layers = [wrapped_model.model.model[layer_idx]]
            with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
                grad_cam_heatmap = cam(input_tensor=input_tensor, targets=targets)[0]
                grad_cam_result = show_cam_on_image(img_float, grad_cam_heatmap, use_rgb=True)
                filename = OUT_DIR.joinpath("grad_cam", str(layer_idx), img_path.name)
                filename.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(filename), cv2.cvtColor(grad_cam_result, cv2.COLOR_RGB2BGR))
                print("Grad-CAM saved to grad_cam_output.jpg")

            print("\n--- Running EigenCAM ---")
            for layer_idx in layers["eigen_cam"]:
                target_layer = [wrapped_model.model.model[layer_idx]]
                with EigenCAM(model=wrapped_model, target_layers=target_layer) as cam:
                    heatmap = cam(input_tensor=input_tensor, targets=None)[0]
                    result = show_cam_on_image(img_float, heatmap, use_rgb=True)

                    filename = OUT_DIR.joinpath("eigen_cam", str(layer_idx), img_path.name)
                    filename.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    print(f"Saved: {filename}")
