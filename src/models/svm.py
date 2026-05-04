import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models.base import BaseDetector
from utils.logger import Logger


class SVMDetector(BaseDetector):
    def __init__(
        self,
        use_pca=True,
        pca_variance=0.85,
        C=0.05,
        kernel="linear",
        threshold=0.5,
        verbose=False,
        class_weight="balanced",
    ):
        self.threshold = threshold
        self._logger = Logger(verbose)

        steps = [("scaler", StandardScaler())]
        if use_pca:
            steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))

        steps.append(
            (
                "svm",
                SVC(
                    C=C,
                    kernel=kernel,
                    class_weight=class_weight,
                    probability=True,
                    random_state=42,
                ),
            )
        )

        self.pipeline = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMDetector":
        self._logger.info(f"  Training SVM. Input dim: {X.shape[1]}, samples: {X.shape[0]}")
        self.pipeline.fit(X, y)

        if "pca" in self.pipeline.named_steps:
            n = self.pipeline.named_steps["pca"].n_components_
            self._logger.info(f"  PCA reduced dim to: {n}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts based on the probability threshold (0.5)
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Returns TRUE probabilities naturally: [prob_class_0, prob_class_1]
        return self.pipeline.predict_proba(X)
