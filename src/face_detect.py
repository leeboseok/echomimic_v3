import sys
import mediapipe as mp
from PIL import Image
import numpy as np


def get_mask_coord(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    height, width, _ = img_np.shape

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    results = detector.process(img_np)
    detector.close()

    if not results.detections:
        print(f'{image_path} has no face detected!')
        return None

    det = results.detections[0]
    box = det.location_data.relative_bounding_box
    x = int(box.xmin * width)
    y = int(box.ymin * height)
    x2 = int((box.xmin + box.width) * width)
    y2 = int((box.ymin + box.height) * height)

    return y, y2, x, x2, height, width


if __name__ == "__main__":
    image_path = sys.argv[1]
    result = get_mask_coord(image_path)
    if result:
        y, y2, x, x2, height, width = result
        print(y, y2, x, x2, height, width)
