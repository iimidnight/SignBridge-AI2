
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "isl_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

# Check that the required files exist
if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        "Make sure isl_model.keras and class_names.json "
        "are in the project root."
    )

# Load the trained model and its class labels
model = keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# Select an image to test
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Choose an ISL hand-sign image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)
root.destroy()

if not image_path:
    print("No image selected. Run the script again.")
    raise SystemExit

# Prepare the image in the same format used for training
image = Image.open(image_path).convert("RGB")
image = image.resize((128, 128))

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

# Get predictions
predictions = model.predict(image_array, verbose=0)[0]
top_indices = np.argsort(predictions)[::-1][:3]

print("\n--- SignBridge AI: Model Test ---")
print("Image selected:", Path(image_path).name)
print("\nTop predictions:")

for index in top_indices:
    label = class_names[index]
    confidence = predictions[index] * 100
    print(f"{label}: {confidence:.2f}%")
