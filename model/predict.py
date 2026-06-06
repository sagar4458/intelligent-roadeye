"""
Intelligent RoadEye — Inference Engine (PyTorch)
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH       = os.path.join(BASE_DIR, "model", "roadeye_vgg16.pth")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")

IMG_SIZE    = 128
PATCH_SIZE  = 128
STRIDE      = 64

CONFIDENCE_THRESHOLD = 0.75

SEVERITY_THRESHOLDS = {
    "low"     : 0.10,
    "moderate": 0.25,
    "high"    : 0.50,
}

COLORS = {
    "crack"  : (0, 0, 220),
    "pothole": (0, 165, 255),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform ─────────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Load Model ────────────────────────────────────────────────────────────────
_model       = None
_class_names = None


def load_roadeye_model():
    global _model, _class_names

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run model/train.py first."
            )

        print("Loading RoadEye VGG16 model...")

        # Load class indices
        with open(CLASS_INDEX_PATH) as f:
            class_to_idx = json.load(f)
        _class_names = {v: k for k, v in class_to_idx.items()}
        num_classes  = len(class_to_idx)

        # Rebuild model architecture
        model = models.vgg16(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Load saved weights
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.to(device)

        _model = model
        print(f"Model loaded. Classes: {_class_names}")
        print(f"Best val accuracy: {checkpoint.get('val_acc', 'N/A')}")

    return _model, _class_names


# ── Predict Single Patch ──────────────────────────────────────────────────────
def predict_patch(patch_bgr: np.ndarray):
    model, class_names = load_roadeye_model()

    # Convert BGR to RGB PIL image
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(patch_rgb)
    tensor    = infer_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            output = model(tensor)
        probs     = torch.softmax(output, dim=1)[0]
        class_idx = int(probs.argmax())
        confidence= float(probs[class_idx])

    label = class_names[class_idx]
    return label, confidence


# ── Sliding Window Detection ──────────────────────────────────────────────────
def detect_on_frame(frame: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    h, w    = frame.shape[:2]
    results = []

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch       = frame[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            label, conf = predict_patch(patch)

            if conf >= confidence_threshold:
                results.append({
                    "label"     : label,
                    "confidence": round(conf, 3),
                    "bbox"      : [x, y, x + PATCH_SIZE, y + PATCH_SIZE],
                })

    return results


# ── Draw Detections ───────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label           = det["label"]
        conf            = det["confidence"]
        color           = COLORS.get(label, (255, 255, 255))

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text        = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output


# ── Surface Condition ─────────────────────────────────────────────────────────
def compute_surface_condition(detections: list, frame_area: int) -> dict:
    if not detections:
        return {"score": 1.0, "level": "Good", "defect_ratio": 0.0}

    defect_area = len(detections) * (PATCH_SIZE ** 2)
    ratio       = min(defect_area / max(frame_area, 1), 1.0)
    score       = round(1.0 - ratio, 3)

    if ratio >= SEVERITY_THRESHOLDS["high"]:
        level = "Critical"
    elif ratio >= SEVERITY_THRESHOLDS["moderate"]:
        level = "Moderate"
    elif ratio >= SEVERITY_THRESHOLDS["low"]:
        level = "Low"
    else:
        level = "Good"

    return {"score": score, "level": level, "defect_ratio": round(ratio, 3)}


# ── Detection Summary ─────────────────────────────────────────────────────────
def summarize_detections(detections: list) -> dict:
    cracks   = [d for d in detections if d["label"] == "crack"]
    potholes = [d for d in detections if d["label"] == "pothole"]
    total    = len(detections)

    avg_conf = round(
        float(np.mean([d["confidence"] for d in detections])), 3
    ) if detections else 0.0

    return {
        "cracks_detected"   : len(cracks),
        "pothole_candidates": len(potholes),
        "total_defects"     : total,
        "avg_confidence"    : avg_conf,
        "crack_pct"         : round(len(cracks)   / total * 100, 1) if total else 0.0,
        "pothole_pct"       : round(len(potholes) / total * 100, 1) if total else 0.0,
    }
