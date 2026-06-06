"""
Intelligent RoadEye — Flask Backend
Run from project root: python backend/app.py
Dashboard: http://localhost:5000
"""

import os
import sys
import cv2
import json
import time
import base64
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

# Add project root to path so model/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.predict import (
    detect_on_frame,
    draw_detections,
    compute_surface_condition,
    summarize_detections,
    load_roadeye_model,
)

# ── App Setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder = os.path.join(BASE_DIR, "frontend"),
    static_folder   = os.path.join(BASE_DIR, "frontend"),
)

UPLOAD_DIR  = os.path.join(BASE_DIR, "data", "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

app.config["UPLOAD_FOLDER"]       = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"]  = 200 * 1024 * 1024   # 200 MB

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}

# ── Global State ──────────────────────────────────────────────────────────────
analysis_state = {
    "running"          : False,
    "frame_id"         : 0,
    "fps"              : 0,
    "detections"       : [],
    "summary"          : {},
    "condition"        : {},
    "history"          : [],
    "crack_trend"      : [],
    "pothole_trend"    : [],
    "patches"          : [],
    "total_area"       : 0.0,
    "inference_time_ms": 0,
}
state_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_patch_b64(patch_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", patch_bgr)
    return base64.b64encode(buf).decode("utf-8")


def process_frame_update(frame: np.ndarray, frame_id: int):
    t0         = time.time()
    detections = detect_on_frame(frame, confidence_threshold=0.75)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    h, w       = frame.shape[:2]
    frame_area = h * w
    condition  = compute_surface_condition(detections, frame_area)
    summary    = summarize_detections(detections)

    patches = []
    for det in detections[:8]:
        x1, y1, x2, y2 = det["bbox"]
        patch = frame[y1:y2, x1:x2]
        if patch.size > 0:
            patches.append({
                "label"     : det["label"],
                "confidence": det["confidence"],
                "b64"       : encode_patch_b64(cv2.resize(patch, (100, 100))),
            })

    with state_lock:
        analysis_state["frame_id"]           = frame_id
        analysis_state["detections"]         = detections
        analysis_state["summary"]            = summary
        analysis_state["condition"]          = condition
        analysis_state["patches"]            = patches
        analysis_state["inference_time_ms"]  = elapsed_ms
        analysis_state["total_area"]         = round(frame_area / 10_000, 1)

        analysis_state["history"].append(condition["score"])
        analysis_state["crack_trend"].append(summary.get("cracks_detected", 0))
        analysis_state["pothole_trend"].append(summary.get("pothole_candidates", 0))

        if len(analysis_state["history"]) > 1500:
            analysis_state["history"].pop(0)
            analysis_state["crack_trend"].pop(0)
            analysis_state["pothole_trend"].pop(0)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    ext = filename.rsplit(".", 1)[1].lower()
    if ext in {"png", "jpg", "jpeg"}:
        return process_image(filepath)
    else:
        return jsonify({"file": filepath, "type": "video", "filename": filename})


def process_image(filepath: str):
    frame = cv2.imread(filepath)
    if frame is None:
        return jsonify({"error": "Could not read image"}), 400

    process_frame_update(frame, frame_id=1)

    with state_lock:
        dets = analysis_state["detections"][:]

    annotated   = draw_detections(frame, dets)
    result_path = os.path.join(RESULTS_DIR, "result.jpg")
    cv2.imwrite(result_path, annotated)

    # Return relative path for browser
    rel_path = "/results/result.jpg"
    return jsonify({"success": True, "result_image": rel_path})


@app.route("/results/<filename>")
def serve_result(filename):
    from flask import send_from_directory
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/state")
def get_state():
    with state_lock:
        return jsonify({
            "frame_id"          : analysis_state["frame_id"],
            "summary"           : analysis_state["summary"],
            "condition"         : analysis_state["condition"],
            "patches"           : analysis_state["patches"],
            "inference_time_ms" : analysis_state["inference_time_ms"],
            "total_area"        : analysis_state["total_area"],
            "history"           : analysis_state["history"][-100:],
            "crack_trend"       : analysis_state["crack_trend"][-100:],
            "pothole_trend"     : analysis_state["pothole_trend"][-100:],
            "running"           : analysis_state["running"],
        })


@app.route("/api/video_feed/<filename>")
def video_feed(filename):
    filepath = os.path.join(UPLOAD_DIR, secure_filename(filename))

    def generate():
        cap      = cv2.VideoCapture(filepath)
        frame_id = 0
        fps_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            process_frame_update(frame, frame_id)

            annotated = draw_detections(frame, analysis_state["detections"])

            now      = time.time()
            fps      = round(1.0 / max(now - fps_time, 1e-6), 1)
            fps_time = now

            with state_lock:
                analysis_state["fps"] = fps

            _, buf = cv2.imencode(".jpg", annotated)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")

        cap.release()
        with state_lock:
            analysis_state["running"] = False

    with state_lock:
        analysis_state["running"] = True

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/model_info")
def model_info():
    return jsonify({
        "model"        : "VGG16",
        "framework"    : "PyTorch",
        "input_size"   : "128 x 128",
        "classes"      : ["Crack", "Pothole"],
        "cuda_enabled" : True,
        "mixed_precision": "float16",
    })


if __name__ == "__main__":
    print("Loading model...")
    load_roadeye_model()
    print("Starting RoadEye dashboard at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
