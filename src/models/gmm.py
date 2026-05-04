import numpy as np
from scipy.special import logsumexp
from scipy.special import expit  # sigmoid for LLR → probability
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.logger import Logger
from models.base import BaseDetector

# ------------------------------------------------------------------ #
# GMM math helpers (unchanged)
# ------------------------------------------------------------------ #


def logpdf_gauss(x, mu, cov):
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
    return logsumexp(
        [np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)],
        axis=0,
    )


def train_gmm_step(x, ws, mus, covs):
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x) / gammasum[:, np.newaxis]
    if covs[0].ndim == 1:
        covs = gamma.dot(x**2) / gammasum[:, np.newaxis] - mus**2
    else:
        covs = np.array(
            [
                (gamma[i] * x.T).dot(x) / gammasum[i] - mus[i][:, np.newaxis].dot(mus[[i]])
                for i in range(len(ws))
            ]
        )

    return ws, mus, covs, tll


# ------------------------------------------------------------------ #
# Detector
# ------------------------------------------------------------------ #


class GMMDetector(BaseDetector):
    """
    GMM-based speaker detector for audio feature vectors.

    Pipeline: StandardScaler → PCA (optional) → GMM (target) vs GMM (non-target)
    Scores via log-likelihood ratio, converted to probabilities via sigmoid.

    Config yaml block expected:
        models:
          audio:
            _target_: models.gmm.GMMDetector
            n_components: 2
            n_iter: 100
            p_target: 0.135
            use_pca: true
            pca_n_components: 15
            covariance_type: "diag"
            reg_covar: 0.001
            verbose: true
    """

    def __init__(
        self,
        n_components: int = 2,
        n_iter: int = 100,
        p_target: float = 0.5,
        use_pca: bool = True,
        pca_n_components: int = 15,
        threshold: float = 0.5,
        verbose: bool = False,
        covariance_type: str = "diag",
        reg_covar: float = 1e-3,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.p_target = p_target
        self.p_nontarget = 1.0 - p_target
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.threshold = threshold
        self.reg_covar = reg_covar
        self._logger = Logger(verbose)

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_n_components, random_state=42) if use_pca else None
        self.model_target = None
        self.model_nontarget = None

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler + PCA on training data and transform."""
        X = self.scaler.fit_transform(X)
        if self.pca is not None:
            X = self.pca.fit_transform(X)
            self._logger.info(f"  PCA reduced dim to: {self.pca.n_components_}")
        return X

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler + PCA."""
        X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMDetector":
        """
        X: (N, feat_dim) — one row per sample (already extracted stats from AudioProcessor)
        y: (N,)          — binary labels
        """
        self._logger.info(f"  Input dim: {X.shape[1]}, samples: {X.shape[0]}")

        X = self._preprocess_fit(X)

        X_target = X[y == 1]
        X_nontarget = X[y == 0]

        self._logger.info(f"  Target samples: {len(X_target)}, Non-target: {len(X_nontarget)}")

        self.model_target = self._train_gmm(X_target, label="Target")
        self.model_nontarget = self._train_gmm(X_nontarget, label="Non-Target")

        return self

    def _train_gmm(self, data: np.ndarray, label: str) -> dict:
        M = self.n_components
        # Initialise means from random data points
        idx = np.random.default_rng(42).choice(len(data), size=M, replace=False)
        mus = data[idx].copy()

        # Apply the explicit self.reg_covar floor during initialization
        covs = [np.var(data, axis=0) + self.reg_covar] * M
        ws = np.ones(M) / M

        for it in range(self.n_iter):
            ws, mus, covs, tll = train_gmm_step(data, ws, mus, covs)

            # Apply the explicit self.reg_covar floor after every EM step to prevent collapse
            covs = [np.maximum(c, self.reg_covar) for c in covs]

            if (it + 1) % 10 == 0:
                self._logger.info(f"    [{label}] iter {it+1}/{self.n_iter}  TLL={tll:.2f}")

        # ── GMM health check ─────────────────────────────────────────────────
        gamma = np.vstack([np.log(w) + logpdf_gauss(data, m, c) for w, m, c in zip(ws, mus, covs)])
        responsibilities = np.exp(gamma - logsumexp(gamma, axis=0))
        effective_counts = responsibilities.sum(axis=1)

        self._logger.info(
            f"  [{label}] Component effective counts: " f"{np.round(effective_counts, 1).tolist()}"
        )
        self._logger.info(f"  [{label}] Component weights: " f"{np.round(ws, 3).tolist()}")
        self._logger.info(
            f"  [{label}] Mean cov norms: " f"{[round(float(np.mean(c)), 4) for c in covs]}"
        )

        return {"ws": ws, "mus": mus, "covs": covs}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _llr(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood ratio for each sample row in X (already preprocessed)."""
        ll_t = np.array(
            [
                np.sum(
                    logpdf_gmm(
                        x.reshape(1, -1),
                        self.model_target["ws"],
                        self.model_target["mus"],
                        self.model_target["covs"],
                    )
                )
                for x in X
            ]
        )
        ll_n = np.array(
            [
                np.sum(
                    logpdf_gmm(
                        x.reshape(1, -1),
                        self.model_nontarget["ws"],
                        self.model_nontarget["mus"],
                        self.model_nontarget["covs"],
                    )
                )
                for x in X
            ]
        )
        return (ll_t + np.log(self.p_target)) - (ll_n + np.log(self.p_nontarget))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 2) array of [p_nontarget, p_target],
        consistent with sklearn convention.
        """
        X = self._preprocess(X)
        llr = self._llr(X)
        p_target = expit(llr)  # sigmoid maps LLR → probability
        return np.column_stack([1 - p_target, p_target])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
