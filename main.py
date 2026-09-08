"""
Face Recognition System using PCA (Eigenfaces) + ANN

Pipeline:
1. Load grayscale face images from DATASET_PATH/<person_name>/*.jpg,
   detect + crop the face, and equalize lighting
2. Split into train/test BEFORE any PCA fitting (avoids data leakage)
3. Fit PCA (mean face + eigenfaces) on TRAIN images only
4. Project train/test images into PCA signature space
5. Standardize signatures
6. Cross-validate several ANN configurations (PCA components K, hidden layer
   size, regularization strength) to find the one that generalizes best
7. Train the best configuration on the full train split and report final
   test accuracy
8. Save the trained model for use in live_recognition.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from pca import PCAFaceRecognizer

DATASET_PATH = "dataset"
IMG_SIZE = (100, 100)
RANDOM_STATE = 42
MODEL_PATH = "face_model.pkl"

# Reused for cropping just the face region out of each training image, so PCA
# doesn't waste components learning background/hair/clothing variation.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(gray_img):
    """
    Detect the largest face in a grayscale image and return the cropped
    region. Falls back to the full image if no face is detected, so a bad
    photo doesn't silently disappear from the dataset.
    """
    detections = FACE_CASCADE.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(detections) == 0:
        return gray_img

    # Pick the largest detected face (helps if a second, smaller face/false
    # positive shows up in the background of a photo)
    x, y, w, h = max(detections, key=lambda box: box[2] * box[3])
    return gray_img[y : y + h, x : x + w]


def load_dataset(dataset_path, img_size):
    """
    Walk DATASET_PATH/<person_name>/<image>, load each image as grayscale,
    crop to the detected face, equalize lighting, resize, and flatten it
    into a feature vector.

    Returns:
        faces (np.ndarray): shape (num_images, img_size[0] * img_size[1])
        labels (np.ndarray): shape (num_images,)
        label_names (dict): label_id -> person_name
    """
    faces = []
    labels = []
    label_names = {}
    label_id = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        label_names[label_id] = person_name

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            face_crop = detect_and_crop_face(img)

            # Histogram equalization reduces the effect of lighting
            # differences between photos, which otherwise dominate the
            # PCA components before identity does.
            face_crop = cv2.equalizeHist(face_crop)

            face_resized = cv2.resize(face_crop, img_size)
            faces.append(face_resized.flatten())
            labels.append(label_id)

        label_id += 1

    return np.array(faces), np.array(labels), label_names


def build_classifier(alpha):
    """Build a probability-producing multilayer perceptron ANN."""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="lbfgs",
        alpha=alpha,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def augment_training_data(faces, labels):
    """Balance classes and add mirrored examples without changing test data."""
    class_ids, class_counts = np.unique(labels, return_counts=True)
    target_count = int(class_counts.max())
    augmented_faces = []
    augmented_labels = []

    for class_id, class_count in zip(class_ids, class_counts):
        class_faces = faces[labels == class_id]
        repeats = target_count - int(class_count)
        if repeats > 0:
            # Repeat complete samples row-wise; np.resize can splice pixels
            # from neighboring images when repeats is not a multiple of the
            # class size, producing invalid synthetic faces.
            sample_indices = np.arange(repeats) % len(class_faces)
            extra = class_faces[sample_indices].copy()
            extra = extra.reshape(repeats, IMG_SIZE[0], IMG_SIZE[1])[:, :, ::-1]
            augmented_faces.append(extra.reshape(repeats, -1))
            augmented_labels.append(np.full(repeats, class_id))

    if not augmented_faces:
        return faces, labels

    return (
        np.concatenate([faces, *augmented_faces]),
        np.concatenate([labels, *augmented_labels]),
    )


def run_pipeline(X_train_raw, X_test_raw, y_train, y_test, k, alpha):
    """
    Fit PCA on training data only, project both sets, scale, train the ANN,
    and return test accuracy plus the fitted objects.
    """
    X_train_augmented, y_train_augmented = augment_training_data(X_train_raw, y_train)

    pca = PCAFaceRecognizer(n_components=k)
    X_train_sig = pca.fit_transform(X_train_augmented)
    X_test_sig = pca.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sig)
    X_test_scaled = scaler.transform(X_test_sig)

    ann = build_classifier(alpha)
    ann.fit(X_train_scaled, y_train_augmented)

    predictions = ann.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy, predictions, pca, scaler, ann


def cross_validate_config(faces, labels, k, alpha, n_splits=5):
    """
    Estimate accuracy for one (K, hidden_layer_sizes, alpha) combination using
    stratified K-fold CV instead of a single train/test split -- much more
    reliable on small datasets, where one split can easily over- or
    under-estimate accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []

    for train_idx, test_idx in skf.split(faces, labels):
        X_train_raw, X_test_raw = faces[train_idx], faces[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        acc, _, _, _, _ = run_pipeline(
            X_train_raw, X_test_raw, y_train, y_test, k, alpha
        )
        fold_accuracies.append(acc)

    return float(np.mean(fold_accuracies))


def main():
    # ---- 1. Load dataset ----
    faces, labels, label_names = load_dataset(DATASET_PATH, IMG_SIZE)

    print("Face Database Shape:", faces.shape)
    print("Labels Shape:", labels.shape)
    print("\nPersons Found:")
    print("Total Images:", len(faces))
    print("Total Persons:", len(label_names))
    for k, v in label_names.items():
        print(k, "->", v)

    # ---- 2. Train/test split FIRST (before PCA is fit on anything) ----
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        faces,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print("\nTraining Samples:", len(X_train_raw))
    print("Testing Samples:", len(X_test_raw))

    # ---- 3. Save mean face (computed from TRAIN data only) for inspection ----
    mean_face_preview = PCAFaceRecognizer(n_components=1).fit(X_train_raw).mean_face
    mean_image = mean_face_preview.reshape(IMG_SIZE)
    cv2.imwrite("mean_face.jpg", mean_image.astype(np.uint8))
    print("Mean face saved as mean_face.jpg")

    # ---- 4. Grid-search K, hidden layer size, and regularization via CV ----
    k_values = [20, 40, 60]
    alpha_values = [0.0001, 0.001, 0.01]

    print("\n--- Cross-validating ANN configurations ---")
    results = []  # list of (k, hidden, alpha, mean_cv_accuracy)

    for k in k_values:
        for alpha in alpha_values:
            mean_acc = cross_validate_config(faces, labels, k, alpha)
            results.append((k, alpha, mean_acc))
            print(
                f"K={k:>3}  alpha={alpha:<7} "
                f"-> CV Accuracy: {mean_acc * 100:.2f}%"
            )

    best_k, best_alpha, best_acc = max(results, key=lambda r: r[2])
    print(
        f"\nBest config: K={best_k}, alpha={best_alpha} "
        f"({best_acc * 100:.2f}% CV accuracy)"
    )

    # Plot best CV accuracy per K (collapsing over hidden/alpha) to visualize
    # the effect of K specifically
    best_acc_per_k = [
        max(acc for k, _, acc in results if k == kv) for kv in k_values
    ]
    plt.plot(k_values, [a * 100 for a in best_acc_per_k], marker="o")
    plt.xlabel("K Value")
    plt.ylabel("Best Cross-Validated Accuracy (%)")
    plt.title("Accuracy vs K (best ANN config per K, 5-fold CV)")
    plt.grid(True)
    plt.savefig("accuracy_vs_k.png")
    plt.show()

    # ---- 5. Train the best configuration on the full train split for a
    #          final, honest test-set accuracy number ----
    K = best_k
    accuracy, predictions, pca, scaler, ann = run_pipeline(
        X_train_raw, X_test_raw, y_train, y_test, K, best_alpha
    )

    print("\nPrediction sample:", predictions[:20])
    print("Actual sample    :", y_test[:20])
    print("\nFinal Test Accuracy:", round(accuracy * 100, 2), "%")

    # ---- 6. Final result summary ----
    print("\n========== FINAL RESULT ==========")
    print("Total Images:", len(faces))
    print("Total Persons:", len(label_names))
    print("Model: PCA + ANN (MLPClassifier)")
    print("PCA Components (K):", K)
    print("Alpha (regularization):", best_alpha)
    print("Training Samples:", len(X_train_raw))
    print("Testing Samples:", len(X_test_raw))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("==================================")

    # ---- 7. Save the trained model for later use (e.g. live recognition) ----
    model_bundle = {
        "pca": pca,
        "scaler": scaler,
        "ann": ann,
        "label_names": label_names,
        "img_size": IMG_SIZE,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nTrained model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()