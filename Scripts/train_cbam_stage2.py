# train_cbam_stage2.py
# Stage 2: Full fine-tuning with all layers unfrozen at a lower learning rate.

from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("cbam_stage2.pt")

    model.train(
        data="dataset_final.yaml",
        epochs=25,
        imgsz=960,
        batch=8,
        lr0=5e-5,
        freeze=0,       # all layers trainable
        name="cbam_stage2",
        exist_ok=True
    )