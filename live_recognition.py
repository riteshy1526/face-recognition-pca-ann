"""
Live webcam face recognition using the trained PCA + ANN model.

Run main.py first to produce face_model.pkl, then run this script.

Controls:
    q -> quit
"""

import cv2
import numpy as np
import joblib

MODEL_PATH = "face_model.pkl"
CONFIDENCE_THRESHOLD = 0.35  # below this, label as "Unknown"

# OpenCV ships a Haar cascade for frontal face detection
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    raise RuntimeError(
        "Haar cascade failed to load. Check your OpenCV install "
        f"(expected file at: {cv2.data.haarcascades}haarcascade_frontalface_default.xml)"
    )


def load_model(model_path):
    bundle = joblib.load(model_path)
    return (
        bundle["pca"],
        bundle["scaler"],
        bundle["ann"],
        bundle["label_names"],
        bundle["img_size"],
    )


def preprocess_face(gray_frame, box, img_size):
    x, y, w, h = box
    face_crop = gray_frame[y : y + h, x : x + w]
    # Must match the training preprocessing in main.py's load_dataset(),
    # otherwise the model sees a different feature distribution live.
    face_crop = cv2.equalizeHist(face_crop)
    face_resized = cv2.resize(face_crop, img_size)
    return face_resized.flatten().reshape(1, -1)


def resolve_identity(probabilities, ann, label_names, threshold):
    best_idx = int(np.argmax(probabilities))
    class_id = int(ann.classes_[best_idx])
    confidence = float(probabilities[best_idx])

    if confidence < threshold:
        return "Unknown", confidence

    return label_names.get(class_id, str(class_id)), confidence


def predict_identity(face_vector, pca, scaler, ann, label_names):
    signature = pca.transform(face_vector)
    signature_scaled = scaler.transform(signature)

    probabilities = ann.predict_proba(signature_scaled)[0]
    return resolve_identity(probabilities, ann, label_names, CONFIDENCE_THRESHOLD)


def main():
    pca, scaler, ann, label_names, img_size = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend if DirectShow isn't available
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.")
        return

    print("Starting live recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_for_detection = cv2.equalizeHist(gray_raw)  # helps in low/uneven lighting

        faces = FACE_CASCADE.detectMultiScale(
            gray_for_detection, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        # Process every detected face in the frame, not only the largest one.
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)

        # Debug: shows in the terminal whether ANY face was detected this frame
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        for box in faces:
            x, y, w, h = box
            face_vector = preprocess_face(gray_raw, box, img_size)
            name, confidence = predict_identity(
                face_vector, pca, scaler, ann, label_names
            )

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

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()