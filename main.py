import os
from unittest import result
import cv2
import numpy as np

DATASET_PATH = "dataset"
IMG_SIZE = (100, 100)

faces = []
labels = []
label_names = {}

label_id = 0

for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_names[label_id] = person_name

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)

        img_vector = img.flatten()

        faces.append(img_vector)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

print("Face Database Shape:", faces.shape)
print("Labels Shape:", labels.shape)

print("\nPersons Found:")
for k, v in label_names.items():
    print(k, "->", v)

    # Mean Face Calculation

mean_face = np.mean(faces, axis=0)

print("\nMean Face Shape:", mean_face.shape)
print("First 10 Mean Values:")
print(mean_face[:10])

mean_image = mean_face.reshape(100, 100)

cv2.imwrite("mean_face.jpg", mean_image)
print("Mean face saved as mean_face.jpg")

# Mean Face Calculation

mean_face = np.mean(faces, axis=0)

print("\nMean Face Shape:", mean_face.shape)
print("First 10 Mean Values:")
print(mean_face[:10])

# Save Mean Face Image

mean_image = mean_face.reshape(100, 100)

cv2.imwrite("mean_face.jpg", mean_image.astype(np.uint8))

print("Mean face saved as mean_face.jpg")
# Mean Zero Data (Delta Matrix)

delta = faces - mean_face

print("\nDelta Matrix Shape:", delta.shape)

# Mean Zero Data (Delta Matrix)

delta = faces - mean_face

print("\nDelta Matrix Shape:", delta.shape)

print("\nFirst Image First 10 Values:")
print(faces[0][:10])

print("\nMean Face First 10 Values:")
print(mean_face[:10])

print("\nDelta First 10 Values:")
print(delta[0][:10])

# Surrogate Covariance Matrix

cov_matrix = np.dot(delta, delta.T)

print("\nCovariance Matrix Shape:", cov_matrix.shape)
# Eigen Decomposition

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("\nEigenvalues Shape:", eigenvalues.shape)
print("Eigenvectors Shape:", eigenvectors.shape)

# Mean Zero Data

delta = faces - mean_face

print("\nDelta Matrix Shape:", delta.shape)

# Surrogate Covariance Matrix

cov_matrix = np.dot(delta, delta.T)

print("Covariance Matrix Shape:", cov_matrix.shape)

# Eigen Decomposition

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("Eigenvalues Shape:", eigenvalues.shape)
print("Eigenvectors Shape:", eigenvectors.shape)
# Sort Eigenvalues in Descending Order

sorted_idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Select Top K Eigenvectors

K = 70

top_eigenvectors = eigenvectors[:, :K]

print("\nTop Eigenvectors Shape:", top_eigenvectors.shape)
# Generate Eigenfaces

eigenfaces = np.dot(delta.T, top_eigenvectors)

# Normalize Eigenfaces

for i in range(K):
    eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])

print("Eigenfaces Shape:", eigenfaces.shape)

# Generate Face Signatures

signatures = np.dot(delta, eigenfaces)

print("\nFace Signatures Shape:", signatures.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    signatures,
    labels,
    test_size=0.4,
    random_state=42,
    stratify=labels
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ann = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=42
)
ann.fit(X_train, y_train)

predictions = ann.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
predictions = ann.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Prediction sample:", predictions[:20])
print("Actual sample    :", y_test[:20])

print("\nAccuracy:", round(accuracy * 100, 2), "%")

#result 
print("\n========== FINAL RESULT ==========")
print("Total Images:", len(faces))
print("Total Persons:", len(label_names))
print("PCA Components (K):", K)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Model: PCA + ANN")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("==================================")
print("Prediction sample:", predictions[:10])
print("Actual sample:", y_test[:10])