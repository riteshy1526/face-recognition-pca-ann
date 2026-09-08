import numpy as np


class PCAFaceRecognizer:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_face = None
        self.eigenfaces = None

    def fit(self, X):
        """
        Fit PCA using training face images only.

        X shape:
        (number_of_images, 10000)
        """

        # Mean face
        self.mean_face = np.mean(X, axis=0)

        # Mean-centered data
        delta = X - self.mean_face

        # Surrogate covariance matrix
        covariance_matrix = np.dot(delta, delta.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select top K eigenvectors
        k = min(self.n_components, len(eigenvalues))

        top_eigenvectors = eigenvectors[:, :k]

        # Generate eigenfaces
        eigenfaces = np.dot(delta.T, top_eigenvectors)

        # Normalize eigenfaces
        for i in range(eigenfaces.shape[1]):
            norm = np.linalg.norm(eigenfaces[:, i])

            if norm > 0:
                eigenfaces[:, i] /= norm

        self.eigenfaces = eigenfaces

        return self

    def transform(self, X):
        """
        Convert face images into PCA signatures.
        """

        if self.mean_face is None or self.eigenfaces is None:
            raise ValueError("PCA model has not been fitted yet.")

        delta = X - self.mean_face

        signatures = np.dot(delta, self.eigenfaces)

        return signatures

    def fit_transform(self, X):
        """
        Fit PCA and transform training data.
        """

        self.fit(X)

        return self.transform(X)