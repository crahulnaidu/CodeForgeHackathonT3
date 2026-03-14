# build_cbam_model.py
# Injects CBAM attention modules into a pre-trained YOLOv8 baseline model
# and saves the modified model for fine-tuning.

from ultralytics import YOLO
import torch.nn as nn
from cbam_module import CBAM

if __name__ == "__main__":

    yolo = YOLO("baseline_trained.pt")

    # Access the internal backbone layer list
    m = yolo.model.model

    # Backbone/neck block indices where CBAM will be inserted
    targets = [4, 6, 8]

    for t in targets:

        block = m[t]

        # Find the output channel count from the last conv layer in this block
        ch = None
        for sub in block.modules():
            if hasattr(sub, "out_channels"):
                ch = sub.out_channels

        if ch is None:
            raise Exception("channel not found")

        print("Injecting CBAM at", t, "channels =", ch)

        m[t] = nn.Sequential(
            block,
            CBAM(ch)
        )

    # Save via Ultralytics API to preserve full model metadata
    yolo.save("yolov8_cbam_init.pt")

    print("Correct CBAM model saved")