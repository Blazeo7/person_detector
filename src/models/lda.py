import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from .base import BaseDetector
from utils.logger import Logger


class LDADetector(BaseDetector):
    def __init__(self, use_pca=True, pca_variance=0.95, verbose=False):
        """
        Args:
            use_pca: Whether to compress features using PCA before LDA.
            pca_variance: Keep enough components to explain this % of the variance.
        """
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self._logger = Logger(verbose)

        # Initialize the sklearn models
        self.pca = PCA(n_components=self.pca_variance) if self.use_pca else None

        # We use SVD solver as it's the most stable without shrinkage
        self.lda = LinearDiscriminantAnalysis(solver="svd")

    def fit(self, X: list, y: np.ndarray):
        X_mat = np.vstack(X)

        self._logger.info(f"Training LDA. Original feature dimension: {X_mat.shape[1]}")

        if self.use_pca:
            X_mat = self.pca.fit_transform(X_mat)
            self._logger.info(
                f"Applied PCA. Reduced dimension to: {X_mat.shape[1]} "
                f"(keeping {self.pca_variance*100}% variance)"
            )

        self.lda.fit(X_mat, y)
        return self

    def predict_score(self, X: list) -> np.ndarray:
        """
        LDA decision_function returns the distance to the separating hyperplane.
        A score > 0 means class 1 (Target), < 0 means class 0 (Non-Target).
        """
        X_mat = np.vstack(X)
        if self.use_pca:
            X_mat = self.pca.transform(X_mat)

        return self.lda.decision_function(X_mat)
