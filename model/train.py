"""
Plant Leaf Disease Detection - CNN Training Script
Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset
Framework: TensorFlow / Keras
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

#  CONFIGURATION

DATASET_DIR   = "../dataset"          # Root folder with Train/ and Test/ subfolders
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
MODEL_SAVE    = "plant_disease_model.h5"
CLASSES_FILE  = "class_names.txt"

#  DATA AUGMENTATION & GENERATORS

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.15,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_dir = os.path.join(DATASET_DIR, "Train")
test_dir  = os.path.join(DATASET_DIR, "Test")

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

NUM_CLASSES  = len(train_gen.class_indices)
CLASS_NAMES  = list(train_gen.class_indices.keys())
print(f"\nClasses found ({NUM_CLASSES}): {CLASS_NAMES}\n")

# Save class names for Flask app
with open(CLASSES_FILE, "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

#  MODEL  —  MobileNetV2 Transfer Learning

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze base initially

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

#  PHASE 1 — Train top layers

callbacks = [
    ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6, monitor="val_loss", verbose=1),
]

print("\nPhase 1: Training top layers ...\n")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

#  PHASE 2 — Fine-tune last 30 layers of base

print("\nPhase 2: Fine-tuning base model ...\n")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
)


#  EVALUATION

print("\nEvaluating on test set ...")
loss, acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc * 100:.2f}%  |  Test Loss: {loss:.4f}")

#  PLOT TRAINING CURVES

def merge_histories(h1, h2, key):
    return h1.history[key] + h2.history[key]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(merge_histories(history1, history2, "accuracy"),      label="Train Acc")
axes[0].plot(merge_histories(history1, history2, "val_accuracy"),  label="Val Acc")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(merge_histories(history1, history2, "loss"),          label="Train Loss")
axes[1].plot(merge_histories(history1, history2, "val_loss"),      label="Val Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nTraining curves saved -> training_curves.png")
print(f"Model saved -> {MODEL_SAVE}")
print(f"Class names saved -> {CLASSES_FILE}")
