import numpy as np
from models.base import BaseDetector
from utils.logger import Logger


def logistic_sigmoid(a):
    a = np.clip(a, -500, 500)
    return 1 / (1 + np.exp(-a))


def train_linear_logistic_regression_GD(x, class_ids, wold, w0old, learning_rate=None):
    if learning_rate is None:
        learning_rate = 0.003 / len(x)

    x = np.c_[x, np.ones(len(x))]
    wold = np.r_[wold, w0old]
    posteriors = logistic_sigmoid(x.dot(wold))
    wnew = wold - learning_rate * (posteriors - class_ids).dot(x)
    return wnew[:-1], wnew[-1]


class ScoreFusionDetector(BaseDetector):
    def __init__(
        self,
        model_audio: BaseDetector = None,
        model_image: BaseDetector = None,
        epochs=200,
        learning_rate=0.05,
        threshold=0.1,
        verbose=False,
    ):
        self.model_audio = model_audio
        self.model_image = model_image
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.verbose = verbose
        self._logger = Logger(verbose)

        # Linear fusion weights
        self.w = np.zeros(2)
        self.w0 = 0.0

    def fit(self, X: list, y: np.ndarray):
        """X must be a list of tuples: [(features_audio, features_image), ...]"""
        X_audio = [item[0] for item in X]
        X_image = [item[1] for item in X]

        self._logger.info("Extracting scores for fusion training...")

        scores_audio = self.model_audio.predict_proba(X_audio)[:, 1]
        scores_image = self.model_image.predict_proba(X_image)[:, 1]

        # Combine into a 2D matrix (N, 2)
        S_mat = np.column_stack((scores_audio, scores_image))

        self._logger.info("Training Logistic Regression on modality scores...")
        for epoch in range(self.epochs):
            self.w, self.w0 = train_linear_logistic_regression_GD(
                S_mat, y, self.w, self.w0, self.learning_rate
            )

            if self.verbose and (epoch + 1) % 50 == 0:
                posteriors = logistic_sigmoid(S_mat.dot(self.w) + self.w0)
                loss = -np.mean(
                    y * np.log(posteriors + 1e-15) + (1 - y) * np.log(1 - posteriors + 1e-15)
                )
                self._logger.info(f"  Epoch {epoch+1}/{self.epochs} - Score Fusion Loss: {loss:.4f}")

        self._logger.info(f"Model weight: {self.w}")
        return self

    def predict_proba(self, X: list) -> np.ndarray:
        """Outputs normalized probabilities bounded between 0 and 1."""
        X_audio = [item[0] for item in X]
        X_image = [item[1] for item in X]

        scores_audio = self.model_audio.predict_proba(X_audio)[:, 1]
        scores_image = self.model_image.predict_proba(X_image)[:, 1]

        S_mat = np.column_stack((scores_audio, scores_image))
        logits = S_mat.dot(self.w) + self.w0

        # Apply sigmoid so it outputs a true probability [0, 1]
        return logistic_sigmoid(logits)

    def predict(self, X: list) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)
