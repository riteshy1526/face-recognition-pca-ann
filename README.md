# Face Recognition Using PCA and ANN

A face recognition system built from scratch with **Principal Component Analysis (Eigenfaces)** for feature extraction and an **Artificial Neural Network (MLPClassifier)** for classification. It includes a training pipeline, a live webcam recognizer, and a small Flask web app.

## Results

| Setting | Classes | Test accuracy |
|---|---|---|
| 40-class face dataset (i-STUDIO AI internship) | 40 | **91%** |

The `dataset/` folder in this repository is a small demo set (4 people: Alok, Nitin, Ritesh, Utkarsh, 114 images) so the project can be run and tested quickly. Accuracy on this tiny set is much lower than on the 40-class dataset because there are very few images per person; add more images per person to improve it.

## Features

- Face detection and cropping with an OpenCV Haar cascade, plus lighting equalization
- Train/test split done **before** fitting PCA, so there is no data leakage
- PCA implemented step by step: mean face, delta matrix, covariance matrix, eigenvalues/eigenvectors, eigenfaces, face signatures
- Cross-validation over the number of PCA components (K), hidden-layer size and regularization to pick the best ANN configuration
- Live webcam recognition with an "Unknown" label when confidence is below 0.35
- Flask web app that streams the webcam feed with recognition boxes in the browser
- Unit tests for the label-mapping logic

## Tech Stack

Python, OpenCV, NumPy, Scikit-learn, Matplotlib, Joblib, Flask

## Project Structure

```text
main.py                 Train: load data, PCA, ANN tuning, evaluate, save model
pca.py                  PCA / Eigenfaces implementation
ann.py                  ANN helpers
live_recognition.py     Live webcam recognition (OpenCV window)
app.py                  Flask web app for live recognition in the browser
utils.py                Shared helpers
tests/                  Unit tests
dataset/<person>/*.jpg  Training images, one folder per person
plots/                  Generated plots (mean face, eigenfaces, etc.)
Report.pdf              Project report
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Put face images in `dataset/<person_name>/` (one folder per person).
2. Train the model (saves `face_model.pkl`):

   ```bash
   python main.py
   ```

3. Run live recognition:

   ```bash
   python live_recognition.py     # OpenCV window, press q to quit
   python app.py                  # browser app at http://127.0.0.1:5000
   ```

> The web app uses the webcam of the machine it runs on and has no authentication, so it is meant for local use.

## Author

Ritesh Kumar Yadav, B.Tech CSE (AI & ML)

## Future Improvements

- Deep learning based face embeddings
- Larger and more varied dataset
- Multi-camera / multi-user web deployment
