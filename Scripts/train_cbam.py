# train_cbam.py
# Fine-tunes the CBAM-enhanced model with frozen backbone layers.

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("cbam_model.pt")

    model.train(
        data="dataset_final.yaml",
        epochs=30,
        imgsz=960,
        batch=8,
        lr0=5e-5,
        freeze=10,      # freeze early backbone layers
        project="runs_cbam",
        name="cbam_finetune"
    )