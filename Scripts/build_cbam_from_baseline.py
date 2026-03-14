# build_cbam_from_baseline.py
# Injects CBAM attention modules into specific backbone/neck blocks of a
# pre-trained YOLOv8 model and saves the raw PyTorch state.

from ultralytics import YOLO
import torch
from cbam_module import CBAM

if __name__ == "__main__":
    # Load baseline model (Ultralytics Model instance)
    model = YOLO("baseline_trained.pt")  # must exist in cwd

    # Access the underlying nn.Module
    m = model.model

    # Backbone/neck block indices to attach CBAM after.
    # These indices may vary with YOLO version/variant.
    targets = [4, 6, 8]  # tweak if needed

    # Resolve the internal layer list (handles different Ultralytics versions)
    try:
        layers = m.model  # m is Model; m.model is the sequential layer list
    except Exception:
        layers = m

    for t in targets:
        if t >= len(layers):
            print(f"skip target {t} (out of range)")
            continue
        block = layers[t]
        # Determine the output channel count from the block's convolutions
        ch = None
        if hasattr(block, "cv2") and hasattr(block.cv2, "conv"):
            ch = block.cv2.conv.out_channels
        elif hasattr(block, "conv") and hasattr(block.conv, "out_channels"):
            ch = block.conv.out_channels if hasattr(block.conv, "out_channels") else None
        elif hasattr(block, "out_channels"):
            ch = block.out_channels
        else:
            # Fallback: scan all sub-modules for the last Conv2d
            for n, sub in block.named_modules():
                if isinstance(sub, torch.nn.Conv2d):
                    ch = sub.out_channels
                    break
        if ch is None:
            raise RuntimeError(f"Could not determine out channels for block {t}")
        print(f"injecting CBAM into block {t} with channels {ch}")
        # Replace the block with Sequential(block, CBAM)
        layers[t] = torch.nn.Sequential(block, CBAM(ch))

    # Save the modified model weights as a raw PyTorch checkpoint
    torch.save(m, "yolov8_cbam_init.pt")
    print("Saved yolov8_cbam_init.pt")