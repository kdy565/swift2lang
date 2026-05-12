import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple


class RidgeRegressionProbe:
    """Ridge regression from SwiFT features -> LLM word embeddings.

    Fits a separate ridge regression for each output dimension, or uses
    a multi-output ridge (sklearn supports multioutput natively).

    Hyperparameter selection via bootstrap or grid search on a held-out
    validation split.
    """

    def __init__(
        self,
        alphas: List[float] = None,
        fit_intercept: bool = True,
        max_iter: int = 1000,
    ):
        if alphas is None:
            alphas = [0.1, 1, 10, 100, 1000, 10000]
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.best_alpha_ = None
        self.model_: Optional[Pipeline] = None

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

    def fit_with_grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.15,
        n_bootstrap: int = 20,
    ) -> Tuple[float, float]:
        """Grid search over alphas using a bootstrap validation procedure.

        Args:
            X: (n_samples, n_features) SwiFT features.
            y: (n_samples, n_target_dims) LLM embeddings.
            val_ratio: fraction of samples for validation.
            n_bootstrap: number of bootstrap rounds.

        Returns:
            (best_alpha, best_val_corr) where corr is mean Pearson across dims.
        """
        n = len(X)
        n_val = max(1, int(n * val_ratio))

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

        # Refit on full data
        self.model_ = self._build_pipeline(self.best_alpha_)
        self.model_.fit(X, y)

        return self.best_alpha_, best_val_corr

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: Optional[float] = None):
        """Fit with a specific alpha (no grid search)."""
        alpha = alpha or (self.alphas[0] if self.alphas else 1.0)
        self.best_alpha_ = alpha
        self.model_ = self._build_pipeline(alpha)
        self.model_.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


def _pearson_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation per output dimension."""
    y_true = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (y_true * y_pred).sum(axis=0)
    den = np.sqrt((y_true ** 2).sum(axis=0) * (y_pred ** 2).sum(axis=0))
    den = np.clip(den, 1e-12, None)
    return num / den
