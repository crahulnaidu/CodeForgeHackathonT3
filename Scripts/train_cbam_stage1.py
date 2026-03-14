# train_cbam_stage1.py
# Stage 1: Fine-tune CBAM-injected model with frozen backbone layers
# so only the CBAM modules and detection head learn initially.

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8_cbam_init.pt")   # must exist
    # Freeze the first 10 layers (backbone) so CBAM + head adjust first
    model.train(data="dataset_final.yaml", epochs=8, imgsz=960, batch=8, freeze=10, lr0=1e-4, name="cbam_stage1", exist_ok=True)