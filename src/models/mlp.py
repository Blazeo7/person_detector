import numpy as np

from utils.logger import Logger
from .base import BaseDetector


def logistic_sigmoid(a):
    # Clipped to prevent overflow warnings in np.exp
    a = np.clip(a, -500, 500)
    return 1 / (1 + np.exp(-a))


def eval_nnet(x, w1, w2):
    h = logistic_sigmoid(np.c_[np.ones(len(x)), x].dot(w1))
    return logistic_sigmoid(np.c_[np.ones(len(h)), h].dot(w2))


def train_nnet(X, T, w1, w2, epsilon):
    mixer = np.random.permutation(len(X))
    X = X[mixer]
    T = T[mixer]
    ed = 0
    for x, t in zip(X, T):
        h = logistic_sigmoid(np.r_[1, x].dot(w1))
        y = logistic_sigmoid(np.r_[1, h].dot(w2))

        de_da2 = y - t
        de_dh = w2[1:] * de_da2

        de_da1 = de_dh * h * (1 - h)

        w1 -= epsilon * np.r_[1, x][:, np.newaxis].dot(de_da1[np.newaxis, :])
        w2 -= epsilon * np.r_[1, h] * de_da2

        # Cross entropy loss calculation
        # Clip y to prevent log(0)
        y_safe = np.clip(y, 1e-15, 1 - 1e-15)
        ed -= t * np.log(y_safe) + (1 - t) * np.log(1 - y_safe)
    return (w1, w2, ed)


class MLPDetector(BaseDetector):
    def __init__(self, hidden_size=50, epochs=100, learning_rate=0.01, verbose=False):
        """
        Args:
            hidden_size: Number of neurons in the hidden layer.
            epochs: Number of training passes over the dataset.
            learning_rate: Epsilon for gradient descent.
            verbose: If True, prints loss during training.
        """
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._logger = Logger(verbose)

        self.w1 = None
        self.w2 = None

    def fit(self, X: list, y: np.ndarray):
        # Stack feature vectors into a 2D matrix
        X_mat = np.vstack(X)
        input_dim = X_mat.shape[1]

        # Initialize weights randomly
        # w1 includes bias row, w2 includes bias scalar
        self.w1 = np.random.randn(input_dim + 1, self.hidden_size) * 0.1
        self.w2 = np.random.randn(self.hidden_size + 1) * 0.1

        self._logger.info(f"Training MLP on {len(X_mat)} samples with dimension {input_dim}...")

        for epoch in range(self.epochs):
            self.w1, self.w2, loss = train_nnet(X_mat, y, self.w1, self.w2, self.learning_rate)
            if (epoch + 1) % 10 == 0:
                self._logger.info(f"  Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")

        return self

    def predict_score(self, X: list) -> np.ndarray:
        """
        Returns log-odds score: log(p / (1-p)).
        This centers the threshold exactly at 0.0, matching the GMM behavior.
        """
        X_mat = np.vstack(X)
        probs = eval_nnet(X_mat, self.w1, self.w2)

        # Clip to prevent division by zero or log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        # Convert to log-odds: log(p / (1-p))
        logits = np.log(probs / (1 - probs))
        return logits
