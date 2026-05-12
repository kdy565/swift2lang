import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple


class VoxelRidgeBaseline:
    """Voxel-wise ridge regression from BOLD -> LLM word embeddings.

    This replicates the Huth et al. / Tang et al. baseline:
    each voxel's BOLD timecourse is used to predict the word embedding
    at each TR. The ridge regularization is tuned independently per
    output dimension (or shared for efficiency).

    For memory efficiency, voxels can be subsampled.
    """

    def __init__(
        self,
        alphas: List[float] = None,
        n_voxel_samples: Optional[int] = 50000,
        fit_intercept: bool = True,
        max_iter: int = 1000,
    ):
        if alphas is None:
            alphas = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
        self.alphas = alphas
        self.n_voxel_samples = n_voxel_samples
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.best_alpha_: Optional[float] = None
        self.model_: Optional[Pipeline] = None
        self.voxel_mask_: Optional[np.ndarray] = None

    def _subsample_voxels(self, X: np.ndarray) -> np.ndarray:
        if self.n_voxel_samples is None or X.shape[1] <= self.n_voxel_samples:
            self.voxel_mask_ = np.ones(X.shape[1], dtype=bool)
            return X
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[1], self.n_voxel_samples, replace=False)
        self.voxel_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.voxel_mask_[idx] = True
        return X[:, idx]

    def _build_pipeline(self, alpha: float) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(
                alpha=alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                random_state=42,
            )),
        ])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.15,
        n_bootstrap: int = 10,
        alpha: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Fit voxel ridge regression.

        Args:
            X: (n_trs, n_voxels) BOLD responses.
            y: (n_trs, n_target_dims) LLM embeddings.
            val_ratio: validation split for hyperparameter selection.
            n_bootstrap: bootstrap rounds for alpha selection.
            alpha: if given, skip grid search and use this alpha.

        Returns:
            (best_alpha, best_val_corr).
        """
        X = self._subsample_voxels(X)
        n = len(X)
        n_val = max(1, int(n * val_ratio))

        if alpha is not None:
            self.best_alpha_ = alpha
            self.model_ = self._build_pipeline(alpha)
            self.model_.fit(X, y)
            return alpha, 0.0

        alpha_scores = {a: [] for a in self.alphas}

        for _ in range(n_bootstrap):
            idx = np.random.permutation(n)
            train_idx, val_idx = idx[n_val:], idx[:n_val]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for alpha in self.alphas:
                pipe = self._build_pipeline(alpha)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_val)
                corr = _pearson_per_dim(y_val, y_pred).mean()
                alpha_scores[alpha].append(corr)

        mean_scores = {a: np.mean(v) for a, v in alpha_scores.items()}
        self.best_alpha_ = max(mean_scores, key=mean_scores.get)
        best_val_corr = mean_scores[self.best_alpha_]

        self.model_ = self._build_pipeline(self.best_alpha_)
        self.model_.fit(X, y)

        return self.best_alpha_, best_val_corr

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voxel_mask_ is not None:
            X = X[:, self.voxel_mask_]
        return self.model_.predict(X)


def _pearson_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (y_true * y_pred).sum(axis=0)
    den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0))
    den = np.clip(den, 1e-12, None)
    return num / den
