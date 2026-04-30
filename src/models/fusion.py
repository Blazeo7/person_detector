import numpy as np
from .base import BaseDetector


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
        model_audio: BaseDetector,
        model_image: BaseDetector,
        epochs=200,
        learning_rate=0.05,
        verbose=False,
    ):
        """
        Args:
            model_audio: An instantiated (and ideally pre-trained) audio model.
            model_image: An instantiated (and ideally pre-trained) image model.
            epochs: Iterations for logistic regression fusion training.
            learning_rate: Gradient descent learning rate.
        """
        self.model_audio = model_audio
        self.model_image = model_image
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Linear fusion weights
        self.w = np.zeros(2)
        self.w0 = 0.0

    def fit(self, X: list, y: np.ndarray):
        """
        X must be a list of tuples: [(features_audio_1, features_image_1), ...]
        """
        X_audio = [item[0] for item in X]
        X_image = [item[1] for item in X]

        if self.verbose:
            print("Extracting scores for fusion training...")

        scores_audio = self.model_audio.predict_score(X_audio)
        scores_image = self.model_image.predict_score(X_image)

        # Combine into a 2D matrix (N, 2)
        S_mat = np.column_stack((scores_audio, scores_image))

        if self.verbose:
            print("Training Logistic Regression on modality scores...")

        for epoch in range(self.epochs):
            self.w, self.w0 = train_linear_logistic_regression_GD(
                S_mat, y, self.w, self.w0, self.learning_rate
            )

            if self.verbose and (epoch + 1) % 50 == 0:
                posteriors = logistic_sigmoid(S_mat.dot(self.w) + self.w0)
                # Compute simple cross-entropy loss for monitoring
                loss = -np.mean(
                    y * np.log(posteriors + 1e-15) + (1 - y) * np.log(1 - posteriors + 1e-15)
                )
                print(f"  Epoch {epoch+1}/{self.epochs} - Score Fusion Loss: {loss:.4f}")

        return self

    def predict_score(self, X: list) -> np.ndarray:
        r"""
        Outputs the raw logit ($S \cdot w + w_0$) of the logistic regression.
        If this logit > 0, the probability is > 0.5.
        """
        X_audio = [item[0] for item in X]
        X_image = [item[1] for item in X]

        scores_audio = self.model_audio.predict_score(X_audio)
        scores_image = self.model_image.predict_score(X_image)

        S_mat = np.column_stack((scores_audio, scores_image))

        # Linear combination of scores based on learned weights
        logits = S_mat.dot(self.w) + self.w0
        return logits
