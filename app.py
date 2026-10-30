
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import os
import json
import base64
import secrets
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "signbridge.db")
# GitHub pretrained ISL alphabet model
GITHUB_MODEL_DIR = os.path.join(BASE_DIR, "github_model")

MODEL_PATH = os.path.join(GITHUB_MODEL_DIR, "isl_model.keras")
HAND_MODEL_PATH = os.path.join(GITHUB_MODEL_DIR, "hand_landmarker.task")
LABELS_PATH = os.path.join(GITHUB_MODEL_DIR, "labels.npy")

# Development key. Set FLASK_SECRET_KEY to a persistent random value
# before deploying the application.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Load trained model
# Load GitHub pretrained ISL alphabet model
model = keras.models.load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH, allow_pickle=True)

# MediaPipe hand landmark detector
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

hand_detector = HandLandmarker.create_from_options(hand_options)


def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as connection:
        # Users table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT
                (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
            )
        """)

        # Existing conversation history table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT
                (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
            )
        """)

        # Add user_id to an existing history table if needed.
        columns = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(conversation_history)"
            ).fetchall()
        ]

        if "user_id" not in columns:
            connection.execute("""
                ALTER TABLE conversation_history
                ADD COLUMN user_id INTEGER REFERENCES users(id)
            """)


# Protect dashboard and API routes
def login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please log in first."}), 401
            return redirect(url_for("login"))

        return route_function(*args, **kwargs)

    return wrapper


# HOME / DASHBOARD
@app.route("/")
@login_required
def home():
    return render_template(
        "index.html",
        user_name=session.get("user_name")
    )


# REGISTER
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        if len(name) > 100:
            flash("Name must be 100 characters or fewer.", "error")
            return render_template("register.html")

        if len(email) > 254 or "@" not in email:
            flash("Please enter a valid email address.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)

        try:
            with get_db_connection() as connection:
                connection.execute("""
                    INSERT INTO users (name, email, password)
                    VALUES (?, ?, ?)
                """, (name, email, hashed_password))

            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")

    return render_template("register.html")


# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute("""
                SELECT id, name, email, password
                FROM users
                WHERE email = ?
            """, (email,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            return redirect(url_for("home"))

        flash("Incorrect email or password.", "error")

    return render_template("login.html")


# LOGOUT
@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))



# IMAGE PREDICTION USING GITHUB ISL MODEL
@app.route("/api/predict", methods=["POST"])
@login_required
def predict_sign():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not isinstance(image_data, str) or not image_data.strip():
        return jsonify({"error": "No image provided."}), 400

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data, validate=True)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image."}), 400

        # Convert BGR to RGB for MediaPipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = hand_detector.detect(mp_image)

        # Do not predict if no hand is visible
        if not result.hand_landmarks:
            return jsonify({
                "prediction": "No sign detected",
                "confidence": 0
            })

        # Extract 21 hand landmarks
        landmarks = result.hand_landmarks[0]

        # Normalize coordinates relative to wrist
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y
        wrist_z = landmarks[0].z

        features = []

        for landmark in landmarks:
            features.extend([
                landmark.x - wrist_x,
                landmark.y - wrist_y,
                landmark.z - wrist_z
            ])

        features = np.array(features, dtype=np.float32)

        max_value = np.max(np.abs(features))

        if max_value > 0:
            features = features / max_value

        features = features.reshape(1, 63)

        # Predict alphabet
        predictions = model.predict(features, verbose=0)[0]

        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index]) * 100

        if confidence < 70:
            return jsonify({
                "prediction": "No sign detected",
                "confidence": round(confidence, 2)
            })

        return jsonify({
            "prediction": str(class_names[best_index]),
            "confidence": round(confidence, 2)
        })

    except Exception:
        app.logger.exception("Sign prediction failed")
        return jsonify({"error": "Could not process this image."}), 500


# GET THIS USER'S CONVERSATION HISTORY
@app.route("/api/history", methods=["GET"])
@login_required
def get_history():
    with get_db_connection() as connection:
        rows = connection.execute("""
            SELECT id, text, created_at
            FROM conversation_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 100
        """, (session["user_id"],)).fetchall()

    return jsonify([dict(row) for row in rows])


# SAVE CONVERSATION FOR THIS USER
@app.route("/api/history", methods=["POST"])
@login_required
def save_history():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide some text."}), 400

    if len(text) > 2000:
        return jsonify({"error": "Text is too long."}), 400

    with get_db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO conversation_history (text, user_id)
            VALUES (?, ?)
        """, (text.strip(), session["user_id"]))

        row = connection.execute("""
            SELECT id, text, created_at
            FROM conversation_history
            WHERE id = ? AND user_id = ?
        """, (cursor.lastrowid, session["user_id"])).fetchone()

    return jsonify(dict(row)), 201


# CLEAR ONLY THIS USER'S HISTORY
@app.route("/api/history", methods=["DELETE"])
@login_required
def delete_history():
    with get_db_connection() as connection:
        connection.execute("""
            DELETE FROM conversation_history
            WHERE user_id = ?
        """, (session["user_id"],))

    return jsonify({"message": "Your history has been cleared."})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
