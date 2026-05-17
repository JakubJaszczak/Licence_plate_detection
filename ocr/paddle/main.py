import json
from collections import defaultdict

import cv2
import cv2 as cv
from paddleocr import PaddleOCR

from ocr.util import get_cropped_ground_truth_images


def main():
    ocr = PaddleOCR(use_textline_orientation=True, lang="en", ocr_version="PP-OCRv5")

    # dataset_img_root = Path.cwd().parent.parent.joinpath("datasets", "licence_plate", "images", "test")
    # cropped_images = get_cropped_images(DetectionModelType.YOLO, "v_002")
    cropped_images = get_cropped_ground_truth_images()
    output = defaultdict(dict)
    for f in sorted(cropped_images):
        img = cv.imread(str(f), cv.IMREAD_UNCHANGED)

        if len(img.shape) < 3:
            print(f"Problematic 2D image found: {f} with shape {img.shape}")
        if len(img.shape) == 2:
            # Convert grayscale 2D to 3D BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        i = int(f.stem)
        orig_filename = f.parent.parent.name
        results = ocr.predict(img)
        text = results[0]["rec_texts"]
        if text:
            output[orig_filename].update({i: text})

    with open("labels_output.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
