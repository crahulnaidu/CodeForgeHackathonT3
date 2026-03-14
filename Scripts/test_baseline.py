# test_baseline.py
# Runs inference on a test image using the trained baseline YOLOv8 model.

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("baseline_trained.pt")

    results = model.predict(
        source="test6.png",   # change to any test image path
        conf=0.1,
        iou=0.5,
        imgsz=1500,
        save=True,
        line_width=2
    )

    print("Baseline inference done")