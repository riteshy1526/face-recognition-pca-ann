"""
Live Face Recognition -- Web App

Serves a local website that shows the live webcam feed with face detection
+ PCA/ANN recognition boxes and names drawn on it, same as
live_recognition.py, but viewable in a browser instead of an OpenCV window.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Note: this runs locally and uses the webcam on the machine running the
script. It is not meant to be deployed publicly as-is (no auth, and only one
camera stream at a time).
"""

import cv2
import joblib
import numpy as np
from flask import Flask, Response, render_template

MODEL_PATH = "face_model.pkl"
CONFIDENCE_THRESHOLD = 0.35

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

app = Flask(__name__)

# ---- Load the trained model once at startup ----
_bundle = joblib.load(MODEL_PATH)
PCA_MODEL = _bundle["pca"]
SCALER = _bundle["scaler"]
ANN = _bundle["ann"]
LABEL_NAMES = _bundle["label_names"]
IMG_SIZE = _bundle["img_size"]


def preprocess_face(gray_frame, box, img_size):
    x, y, w, h = box
    face_crop = gray_frame[y : y + h, x : x + w]
    face_crop = cv2.equalizeHist(face_crop)  # match training preprocessing
    face_resized = cv2.resize(face_crop, img_size)
    return face_resized.flatten().reshape(1, -1)


def resolve_identity(probabilities, threshold=CONFIDENCE_THRESHOLD):
    best_idx = int(np.argmax(probabilities))
    class_id = int(ANN.classes_[best_idx])
    confidence = float(probabilities[best_idx])

    if confidence < threshold:
        return "Unknown", confidence

    return LABEL_NAMES.get(class_id, str(class_id)), confidence


def predict_identity(face_vector):
    signature = PCA_MODEL.transform(face_vector)
    signature_scaled = SCALER.transform(signature)

    probabilities = ANN.predict_proba(signature_scaled)[0]
    return resolve_identity(probabilities)


def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_for_detection = cv2.equalizeHist(gray_raw)

        faces = FACE_CASCADE.detectMultiScale(
            gray_for_detection, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        # Process every detected face so the browser stream supports groups.
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        for box in faces:
            x, y, w, h = box
            face_vector = preprocess_face(gray_raw, box, IMG_SIZE)
            name, confidence = predict_identity(face_vector)

            label = f"{name} ({confidence * 100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/")
def index():
    return render_template("index.html", people=list(LABEL_NAMES.values()))


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)