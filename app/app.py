"""
Plant Leaf Disease Detection — Flask Web App
"""

import os
import uuid
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf


#  CONFIG

BASE_DIR       = os.path.dirname(__file__)
MODEL_PATH     = os.path.join(BASE_DIR, "..", "model", "plant_disease_model.h5")
CLASSES_PATH   = os.path.join(BASE_DIR, "..", "model", "class_names.txt")
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT    = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE       = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


#  LOAD MODEL & CLASSES

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(CLASSES_PATH) as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# Disease info map — extend as needed to match your dataset classes
DISEASE_INFO = {
    "Healthy": {
        "status": "healthy",
        "description": "The leaf appears healthy with no signs of disease.",
        "treatment": "No treatment needed. Keep up regular watering and fertilization.",
        "severity": "None",
    },
    "Powdery": {
        "status": "diseased",
        "description": "Powdery mildew is a fungal disease causing white powdery spots on leaves.",
        "treatment": "Apply fungicide (neem oil or sulfur-based). Remove affected leaves. Improve air circulation.",
        "severity": "Moderate",
    },
    "Rust": {
        "status": "diseased",
        "description": "Rust is a fungal disease identified by orange-brown pustules on the leaf surface.",
        "treatment": "Apply copper-based fungicide. Remove and destroy infected leaves. Avoid overhead watering.",
        "severity": "High",
    },
}

# Fallback for unknown class names
DEFAULT_INFO = {
    "status": "unknown",
    "description": "Disease detected. Consult an agricultural expert for proper diagnosis.",
    "treatment": "Isolate the plant and seek professional advice.",
    "severity": "Unknown",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


#  ROUTES

@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, or WEBP."}), 400

    # Save uploaded file with a unique name
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run inference
    img_arr      = preprocess_image(filepath)
    predictions  = model.predict(img_arr)[0]
    top_idx      = int(np.argmax(predictions))
    confidence   = float(predictions[top_idx]) * 100
    class_name   = CLASS_NAMES[top_idx]

    # Top-3 predictions
    top3_idx  = np.argsort(predictions)[::-1][:3]
    top3      = [
        {"class": CLASS_NAMES[i], "confidence": round(float(predictions[i]) * 100, 2)}
        for i in top3_idx
    ]

    info = DISEASE_INFO.get(class_name, DEFAULT_INFO)

    return jsonify({
        "prediction":  class_name,
        "confidence":  round(confidence, 2),
        "top3":        top3,
        "image_url":   url_for("static", filename=f"uploads/{filename}"),
        "info":        info,
        "all_classes": CLASS_NAMES,
        "all_scores":  [round(float(p) * 100, 2) for p in predictions],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES, "model": MODEL_PATH})

#  ENTRY POINT

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
