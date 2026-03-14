import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "cbam_stage2.pt"
IMG_PATH = "test6.png"
OUT_PATH = "gradcam_result.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load YOLO model and enable gradient tracking for GradCAM
yolo = YOLO(MODEL_PATH)
torch_model = yolo.model.to(device)
torch_model.eval()

for param in torch_model.parameters():
    param.requires_grad = True

# Target a deep convolutional layer in the backbone
target_layer = torch_model.model[6].cv2.conv

# 2. Load image and pad dimensions to multiples of 32 (YOLO stride requirement)
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMG_PATH}")

orig_h, orig_w = img_bgr.shape[:2]
pad_h = math.ceil(orig_h / 32) * 32
pad_w = math.ceil(orig_w / 32) * 32

# Pad bottom and right edges with neutral gray to avoid shifting image coordinates
padded_bgr = cv2.copyMakeBorder(
    img_bgr, 
    top=0, bottom=pad_h - orig_h, 
    left=0, right=pad_w - orig_w, 
    borderType=cv2.BORDER_CONSTANT, 
    value=(114, 114, 114)
)

img_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
img_norm = img_rgb / 255.0

input_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
input_tensor.requires_grad_(True)

# 3. Run inference on the padded image to extract the target class
results = yolo(padded_bgr, conf=0.15)
boxes = results[0].boxes

if boxes is None or len(boxes) == 0:
    print("No detections found.")
    exit()

# Identify the class ID of the highest-confidence detection
conf = boxes.conf.cpu().numpy()
best_idx = np.argmax(conf)
target_cls_id = int(boxes.cls[best_idx].item())

# 4. Define the target function for GradCAM
class YOLOTarget:
    def __init__(self, cls_id):
        self.cls_id = cls_id
        
    def __call__(self, model_out):
        # Sum activations for the specified class across all spatial grids
        return model_out[..., self.cls_id].sum()

targets = [YOLOTarget(target_cls_id)]
cam = GradCAM(model=torch_model, target_layers=[target_layer])

# 5. Generate heatmap, crop back to original size, and save
with torch.enable_grad():
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

cam_img_padded = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

# Remove the padding added in step 2 to restore original resolution
cam_img_original_size = cam_img_padded[:orig_h, :orig_w]

cv2.imwrite(OUT_PATH, cv2.cvtColor(cam_img_original_size, cv2.COLOR_RGB2BGR))
print(f"GradCAM saved at original resolution ({orig_w}x{orig_h}) -> {OUT_PATH}")