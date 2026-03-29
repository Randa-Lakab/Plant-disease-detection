# Plant Leaf Disease Detection — CNN + Flask

Detect plant leaf diseases from images using a fine-tuned **MobileNetV2** CNN, served via a **Flask** web app.

---

## Project Structure

```
plant-disease-detection/
├── dataset/
│   ├── Train/
│   │   ├── Healthy/
│   │   ├── Powdery/
│   │   └── Rust/
│   └── Test/
│       ├── Healthy/
│       ├── Powdery/
│       └── Rust/
│
├── model/
│   ├── train.py                  <- CNN training script
│   ├── plant_disease_model.h5    <- saved after training
│   └── class_names.txt           <- auto-generated class list
│
├── app/
│   ├── app.py                    <- Flask backend
│   ├── templates/
│   │   └── index.html            <- Web UI
│   └── static/
│       └── uploads/              <- uploaded images (auto-created)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
cd plant-disease-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

- Go to: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset
- Download and extract into `dataset/`
- Make sure you have `dataset/Train/` and `dataset/Test/` folders

---

## Train the Model

```bash
cd model
python train.py
```

This will:
- Load and augment images from `dataset/Train/`
- Train MobileNetV2 with transfer learning (2 phases)
- Save `plant_disease_model.h5` and `class_names.txt`
- Plot training curves -> `training_curves.png`

> Training takes ~5-15 min on GPU, longer on CPU.

---

## Run the Flask App

```bash
cd app
python app.py
```

Open your browser at: **http://localhost:5000**

- Upload a leaf image (PNG / JPG / WEBP)
- Click **Analyze Leaf**
- See: predicted disease, confidence score, treatment advice, and top-3 predictions

---

## Model Architecture

| Component         | Details                                                 |
|-------------------|---------------------------------------------------------|
| Base Model        | MobileNetV2 (ImageNet pretrained)                       |
| Input Size        | 224 x 224 x 3                                           |
| Head              | GAP -> BN -> Dense(256) -> Dropout -> Dense(128) -> Softmax |
| Training Phase 1  | Frozen base, train head (Adam lr=1e-3)                  |
| Training Phase 2  | Fine-tune last 30 layers (Adam lr=1e-4)                 |
| Augmentation      | Rotation, flip, zoom, shift                             |

---

## Tips

- **Better accuracy**: increase `EPOCHS` or use a larger base model (EfficientNetB3).
- **New classes**: just add more folders inside `dataset/Train/` — the script auto-detects them.
- **GPU**: install `tensorflow-gpu` for much faster training.
- **Production**: swap `app.run(debug=True)` for Gunicorn + Nginx.

---

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- Flask 3
- Pillow
- HTML / CSS / Vanilla JS 
