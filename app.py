
from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Database file will be created in the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "signbridge.db")
MODEL_PATH = os.path.join(BASE_DIR, "isl_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT
                (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
        )
    """)

    connection.commit()
    connection.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_sign():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not isinstance(image_data, str) or not image_data.strip():
        return jsonify({"error": "No image provided."}), 400

    try:
        import base64

        # Remove the data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data, validate=True)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image."}), 400

        # OpenCV uses BGR; the model was trained on RGB images
        debug_path = os.path.join(BASE_DIR, "webcam_debug.jpg")
        cv2.imwrite(debug_path, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frame = frame.astype("float32")
        frame = np.expand_dims(frame, axis=0)

        predictions = model.predict(frame, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index]) * 100

        if confidence < 80:
           return jsonify({
              "prediction": "No sign detected",
              "confidence": round(confidence, 2)
           })

        return jsonify({
    "prediction": class_names[best_index],
    "confidence": round(confidence, 2)
})

        return jsonify({
            "prediction": class_names[best_index],
            "confidence": round(float(predictions[best_index]) * 100, 2)
        })

    except Exception:
        app.logger.exception("Sign prediction failed")
        return jsonify({"error": "Could not process this image."}), 500


# Get saved conversations
@app.route("/api/history", methods=["GET"])
def get_history():
    connection = get_db_connection()

    rows = connection.execute("""
        SELECT id, text, created_at
        FROM conversation_history
        ORDER BY id DESC
        LIMIT 100
    """).fetchall()

    history = [dict(row) for row in rows]
    connection.close()

    return jsonify(history)


# Save a conversation
@app.route("/api/history", methods=["POST"])
def save_history():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide some text."}), 400

    if len(text) > 2000:
        return jsonify({"error": "Text is too long."}), 400

    connection = get_db_connection()

    cursor = connection.execute(
        "INSERT INTO conversation_history (text) VALUES (?)",
        (text.strip(),)
    )

    connection.commit()

    row = connection.execute(
        "SELECT id, text, created_at FROM conversation_history WHERE id = ?",
        (cursor.lastrowid,)
    ).fetchone()

    result = dict(row)
    connection.close()

    return jsonify(result), 201


# Clear all saved conversations
@app.route("/api/history", methods=["DELETE"])
def delete_history():
    connection = get_db_connection()
    connection.execute("DELETE FROM conversation_history")
    connection.commit()
    connection.close()

    return jsonify({"message": "History cleared successfully."})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)


