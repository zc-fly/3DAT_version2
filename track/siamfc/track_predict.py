import glob
import os
import cv2
import sys
from PIL import Image

import torch

sys.path.append(os.getcwd())

from tracker import SiamFCTracker


def track_predict(image):
    gpu_id=0
    model_path='../models/siamfc_pretrained.pth'
    tracker = SiamFCTracker(model_path, gpu_id)

    if idx == 0:
        bbox = detect_startframe(detect_model, frame, device)
        initbbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
        tracker.init(frame, initbbox)

    else:
        bbox = tracker.update(frame)
        # expand bounding box
        bbox = (bbox[0] - 10, bbox[1] - 20,
                bbox[2] + 10, bbox[3] + 10)



if __name__ == '__main__':
    # load image
    img = Image.open("test/004545.jpg").convert('RGB')
    box = track_predict(img)