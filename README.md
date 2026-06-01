# Face Recognition Using PCA and ANN

## Overview

This project implements a Face Recognition System using Principal Component Analysis (PCA) and Artificial Neural Network (ANN). The system extracts important facial features using PCA (Eigenfaces) and classifies individuals using an ANN classifier.

## Features

* Face image preprocessing
* Mean Face Calculation
* Eigenface Generation using PCA
* Dimensionality Reduction
* Face Signature Extraction
* Face Classification using ANN
* Accuracy Evaluation

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn

## Project Workflow

1. Load face images from dataset
2. Convert images to grayscale and resize
3. Generate face database
4. Calculate Mean Face
5. Create Delta Matrix
6. Compute Covariance Matrix
7. Calculate Eigenvalues and Eigenvectors
8. Generate Eigenfaces
9. Extract PCA Features (Face Signatures)
10. Train ANN Classifier
11. Evaluate Recognition Accuracy

## Dataset

The dataset contains images of four individuals:

* Alok
* Nitin
* Ritesh
* Utkarsh

Total Images: 114

## Results

* PCA Components (K): 70
* Training Samples: 68
* Testing Samples: 46
* Classification Model: ANN (MLPClassifier)
* Accuracy Achieved: 39.13%

## Installation

```bash
pip install numpy opencv-python scikit-learn
```

## Run Project

```bash
python main.py
```

## Author

Ritesh Yadav

B.Tech CSE (AI & ML)

## Future Improvements

* Real-time face recognition using webcam
* Face detection before recognition
* Deep Learning based face embeddings
* Improved dataset and model accuracy
