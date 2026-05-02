import numpy as np
from scipy.special import logsumexp

from utils.logger import Logger
from .base import BaseDetector


def logpdf_gauss(x, mu, cov):
    assert mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1])
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5 * (len(mu) * np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum((x**2) / cov, axis=1))
    else:
        return -0.5 * (
            len(mu) * np.log(2 * np.pi)
            + np.linalg.slogdet(cov)[1]
            + np.sum(x.dot(np.linalg.inv(cov)) * x, axis=1)
        )


def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)


def train_gmm_step(x, ws, mus, covs):
    """Single EM iteration for GMM training."""
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x) / gammasum[:, np.newaxis]

    if covs[0].ndim == 1:  # diagonal covariance matrices
        covs = gamma.dot(x**2) / gammasum[:, np.newaxis] - mus**2
    else:
        covs = np.array(
            [
                (gamma[i] * x.T).dot(x) / gammasum[i] - mus[i][:, np.newaxis].dot(mus[[i]])
                for i in range(len(ws))
            ]
        )
    return ws, mus, covs, tll


class GMMDetector(BaseDetector):
    def __init__(self, n_components=30, n_iter=40, p_target=0.5, verbose=False):
        """
        Args:
            n_components: Number of Gaussian mixtures (M).
            n_iter: Number of EM algorithm iterations.
            p_target: Apriori probability of the target (assignment specifies 0.5).
            verbose: If True, prints log-likelihood during training.
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.p_target = p_target
        self.p_nontarget = 1.0 - p_target
        self._logger = Logger(verbose)

        # Dictionaries to hold the trained parameters for both classes
        self.model_target = None
        self.model_nontarget = None

    def fit(self, X: list, y: np.ndarray):
        """
        Trains two separate GMMs: one for target, one for non-target.
        Audio samples are typically sequences of frames, so we stack them
        together before running the EM algorithm.
        """
        # Separate features by class
        X_t = [x for x, label in zip(X, y) if label == 1]
        X_n = [x for x, label in zip(X, y) if label == 0]

        # Stack all frames vertically into a single large dataset for GMM
        train_t = np.vstack(X_t)
        train_n = np.vstack(X_n)

        self._logger.info(f"Training Target GMM on {len(train_t)} frames...")
        self.model_target = self._train_single_gmm(train_t, "Target")

        self._logger.info(f"Training Non-Target GMM on {len(train_n)} frames...")
        self.model_nontarget = self._train_single_gmm(train_n, "Non-Target")

        return self

    def _train_single_gmm(self, data: np.ndarray, label_name: str) -> dict:
        """Internal helper to train a single GMM."""
        M = self.n_components
        mus = data[np.random.randint(1, len(data), M)]
        covs = [np.var(data, axis=0)] * M
        ws = np.ones(M) / M

        for jj in range(self.n_iter):
            ws, mus, covs, tll = train_gmm_step(data, ws, mus, covs)
            if (jj + 1) % 10 == 0:
                self._logger.info(f"  Iteration {jj+1}/{self.n_iter} - TLL ({label_name}): {tll:.2f}")

        return {"ws": ws, "mus": mus, "covs": covs}

    def predict_score(self, X: list) -> np.ndarray:
        """
        Calculates the log-likelihood ratio for each audio sequence.
        """
        scores = []
        for x in X:
            # Calculate log probabilities frame-by-frame
            ll_t = logpdf_gmm(
                x, self.model_target["ws"], self.model_target["mus"], self.model_target["covs"]
            )
            ll_n = logpdf_gmm(
                x, self.model_nontarget["ws"], self.model_nontarget["mus"], self.model_nontarget["covs"]
            )

            # Sum over all frames to get the total log-likelihood for the sequence,
            # then apply Bayes' rule with apriori probabilities.
            score = (np.sum(ll_t) + np.log(self.p_target)) - (np.sum(ll_n) + np.log(self.p_nontarget))
            scores.append(score)

        return np.array(scores)
