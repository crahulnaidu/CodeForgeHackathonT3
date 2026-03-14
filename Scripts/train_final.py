# train_final.py
# Trains the baseline YOLOv8s model on the multiclass dataset.

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset_final.yaml",
        epochs=30,
        imgsz=960,
        batch=8,
        lr0=1e-4,
        freeze=10,          # freeze early backbone layers for transfer learning
        project="runs_final",
        name="baseline_multiclass",
        workers=4
    )