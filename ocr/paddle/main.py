import json
from collections import defaultdict

import cv2 as cv
from pathlib import Path
from paddleocr import PaddleOCR

def main():

    ocr = PaddleOCR(use_textline_orientation=True, lang='en', ocr_version="PP-OCRv5")

    # dataset_img_root = Path.cwd().parent.parent.joinpath("datasets", "licence_plate", "images", "test")
    cropped_images = Path.cwd().parent.parent.joinpath("runs/detect/predict5/license_plate")
    output = defaultdict()
    for f in sorted(cropped_images.iterdir()):
        img = cv.imread(str(f), cv.IMREAD_UNCHANGED)
        results = ocr.predict(img)
        text = results[0]['rec_texts']
        if text:
            output[f.stem] = text

    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)



if __name__ == '__main__':
    main()