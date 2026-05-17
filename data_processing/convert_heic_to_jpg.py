import glob
import os
import shutil
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from data_processing.IoU_calculation import DetectionModelType

register_heif_opener()

output_dir = Path(r"D:\praca_magisterska\project\Licence_plate_detection\datasets\my_dataset\test")
output_dir.mkdir(parents=True, exist_ok=True)


def generate_name():
    for i in range(100000):
        yield i


name_generator = generate_name()


def batch_convert_folder(folder_path):
    # Find all .heic and .HEIC files in the directory
    heic_files = glob.glob(os.path.join(folder_path, "*.heic")) + glob.glob(os.path.join(folder_path, "*.HEIC"))

    if not heic_files:
        print("No HEIC files found in the directory.")
        return

    for i, file_path in enumerate(heic_files):
        output_path = output_dir.joinpath(f"{next(name_generator)}.jpg")

        try:
            image = Image.open(file_path)
            image = image.convert("RGB")
            image.save(output_path, "JPEG")
            print(f"Converted: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to convert {os.path.basename(file_path)}: {e}")


def copy_jpg_files(folder_path):
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.jpeg"))
    for f in jpg_files:
        filename = next(name_generator)
        try:
            # copy2 preserves file metadata
            shutil.copy2(f, output_dir.joinpath(f"{filename}.jpg"))
            print(f"Copied: {Path(f).name} -> {filename}.jpg")
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")


def copy_png_files(output):
    files = Path(
        rf"D:\praca_magisterska\project\Licence_plate_detection\detection\results\{DetectionModelType.DETR.value}\my_dataset_detr_v1"
    ).rglob("results.png")
    output.mkdir(parents=True, exist_ok=True)
    for f in files:
        filename = next(name_generator)
        try:
            # copy2 preserves file metadata
            shutil.copy2(f, output.joinpath(f"{filename}.png"))
            print(f"Copied: {Path(f).name} -> {filename}.png")
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")


if __name__ == "__main__":
    # folder_directory = r"D:\praca_magisterska\project\photos_raw"
    # batch_convert_folder(folder_directory)
    # copy_jpg_files(folder_directory)
    copy_png_files(Path(r"D:\praca_magisterska\project\photos_raw\results\DETR_V1"))
