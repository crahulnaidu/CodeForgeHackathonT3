# multiscale_detect.py
# Runs YOLOv8 inference at multiple image scales and merges detections via NMS
# to improve recall on objects at varying sizes.

import cv2
import numpy as np
from ultralytics import YOLO
import os

# Configuration
MODEL_PATH = "baseline_trained.pt"
IMAGE_PATH = "test5.png"

SCALES = [640, 800, 960, 1280, 1536]

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Read input image
orig = cv2.imread(IMAGE_PATH)
h0, w0 = orig.shape[:2]

all_boxes = []
all_scores = []
all_cls = []

# Run inference at each scale and collect all detections
for s in SCALES:

    results = model.predict(
        source=orig,
        imgsz=s,
        conf=0.15,
        iou=0.5,
        verbose=False
    )

    r = results[0]

    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    for b, sc, c in zip(boxes, scores, cls):
        all_boxes.append(b)
        all_scores.append(sc)
        all_cls.append(int(c))

# Merge overlapping detections across scales using NMS
indices = cv2.dnn.NMSBoxes(
    bboxes=[list(map(int, b)) for b in all_boxes],
    scores=all_scores,
    score_threshold=0.15,
    nms_threshold=0.45
)

final = orig.copy()

if len(indices) > 0:
    for i in indices.flatten():

        x1, y1, x2, y2 = map(int, all_boxes[i])
        sc = all_scores[i]
        c = all_cls[i]

        label = f"{model.names[c]} {sc:.2f}"

        cv2.rectangle(final, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            final,
            label,
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

# Save annotated output image
out_path = os.path.join(OUT_DIR, "multiscale_test6.png")
cv2.imwrite(out_path, final)

print("Saved at:", out_path)