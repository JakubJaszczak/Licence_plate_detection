from pathlib import Path

import cv2


def get_cropped_ground_truth_images():
    return Path(R"D:\praca_magisterska\project\Licence_plate_detection\datasets\cropped_test_images").rglob("*.png")


def get_cropped_images(detection_model_type, version: str):
    root_dir = Path(
        rf"D:\praca_magisterska\project\Licence_plate_detection\detection\results\{detection_model_type.value}\{version}"
    )
    cropped_images = root_dir.rglob("*.jpg")
    return cropped_images


def open_and_wait(image_path):
    image_name = image_path.name
    # Verify the file format is jpg
    if not image_name.lower().endswith(".jpg"):
        print(f"Note: The file '{image_name}' does not have a .jpg extension.")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not find or open the image at '{image_path}'")
        return
    print(f"Successfully loaded image: {image_name}")
    window_title = f"Viewing: {image_name}"
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(" ---------- Image closed.")
