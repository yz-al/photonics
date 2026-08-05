"""
OOD-aware, calibrated predictive uncertainty for the Phase-1 surrogate.

The bare bootstrap-ensemble spread (`ensemble_predict` std) has two documented
failures for active-learning acquisition:

  1. It is ~2x under-dispersed in-distribution (observed RMSE ≈ 2σ).
  2. It is *blind* to out-of-distribution (OOD) inputs: GaAs, a 3D material far
     from the 2D-monolayer training set, was predicted 0.283 ± 0.078 eV — the
     worst error in the study got one of the *lowest* uncertainties.

This module fixes both with two composable, defensible pieces:

  * ``OODDetector`` — a distance-to-training score in standardized feature
    space (kNN mean distance, k=5; Mahalanobis option). Higher = more novel.
    This *sees* extrapolation that an ensemble trained on one distribution
    cannot.
  * ``ConformalCalibrator`` — split-conformal scaling of the ensemble σ so that
    predicted intervals hit nominal coverage (68% / 90%) on held-out data.

``predictive_uncertainty`` combines them:

    σ_total = σ_cal * (1 + alpha * ood_norm)

where ``σ_cal`` is the conformally-calibrated ensemble σ and ``ood_norm`` in
[0, 1] is the training-referenced percentile of the OOD distance. In-
distribution points (ood_norm ≈ 0) keep their calibrated σ; a maximally novel
point (ood_norm ≈ 1) has its σ inflated by ``1 + alpha``. The combination is a
heuristic — a monotone inflation, not a probabilistic OOD posterior — chosen to
be transparent and hard to game (see module-level LIMITATIONS note below).

LIMITATIONS
    Distance-based OOD is a heuristic proxy for epistemic risk; it flags novelty
    in *feature* space, not error in the target. True calibration on the target
    distribution (novel chemistries) needs held-out GW-BSE labels there, which do
    not exist yet. Until then this gives a defensible, monotone acquisition
    signal that is no longer blind to extrapolation — not a certified interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# OOD distance-to-training score
# --------------------------------------------------------------------------- #
@dataclass
class OODDetector:
    """Distance-to-training-set novelty score in standardized feature space.

    Fit on the training feature matrix; score any new matrix. Features are
    median-imputed (training medians) and standardized (training mean/std) so
    every dimension contributes comparably and NaNs (sparse 2D descriptors,
    undefined-for-3D columns) are handled without dropping columns.

    Two scores, both "higher = more OOD":
      * ``method="knn"``    — mean Euclidean distance to the k nearest training
        points in standardized space. Robust, local, no covariance assumptions.
      * ``method="mahalanobis"`` — Mahalanobis distance to the training centroid
        using a (regularized) training covariance. Cheap, global, but assumes an
        ellipsoidal training cloud.
    """

    k: int = 5
    method: str = "knn"
    _median: np.ndarray | None = None
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None
    _train_z: np.ndarray | None = None
    _cov_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "OODDetector":
        """Learn imputation medians, standardization, and the training cloud."""
        X = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore"):
            self._median = np.nanmedian(X, axis=0)
        self._median = np.where(np.isfinite(self._median), self._median, 0.0)
        Xi = self._impute(X)
        self._mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        self._std = np.where(std > 0, std, 1.0)  # guard zero-variance columns
        self._train_z = (Xi - self._mean) / self._std
        # Regularized inverse covariance for the Mahalanobis option.
        cov = np.cov(self._train_z, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        self._cov_inv = np.linalg.pinv(cov)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        assert self._median is not None
        X = np.asarray(X, dtype=float).copy()
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(self._median, idx[1])
        return X

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (self._impute(X) - self._mean) / self._std

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an OOD distance for each row of ``X`` (higher = more novel)."""
        if self._train_z is None:
            raise RuntimeError("OODDetector.score called before fit")
        Z = np.atleast_2d(self._standardize(X))
        if self.method == "mahalanobis":
            # Z is already standardized (training mean removed), so its distance
            # to the training centroid is sqrt(Zᵀ Σ⁻¹ Z).
            m = np.einsum("ij,jk,ik->i", Z, self._cov_inv, Z)
            return np.sqrt(np.maximum(m, 0.0))
        # kNN mean distance (default).
        out = np.empty(Z.shape[0], dtype=float)
        for i, z in enumerate(Z):
            d = np.sqrt(((self._train_z - z) ** 2).sum(axis=1))
            d.sort()
            # If a query coincides with training rows (self-scoring), the first
            # entries are ~0; take the k smallest either way.
            out[i] = d[: self.k].mean()
        return out

    def percentile(self, scores: np.ndarray) -> np.ndarray:
        """Map raw OOD scores to [0, 1] via their rank in the *training* scores.

        1.0 means "at or beyond the most-novel training point"; this is the
        ``ood_norm`` used by :func:`predictive_uncertainty`.
        """
        if self._train_z is None:
            raise RuntimeError("OODDetector.percentile called before fit")
        ref = np.sort(self._self_scores())
        ranks = np.searchsorted(ref, np.asarray(scores, dtype=float), side="right")
        return ranks / len(ref)

    def _self_scores(self) -> np.ndarray:
        """Training-set OOD scores against itself, using the configured method.

        Serves as the reference distribution for :meth:`percentile`. For the kNN
        method the point's own zero self-distance is excluded.
        """
        assert self._train_z is not None
        Z = self._train_z
        if self.method == "mahalanobis":
            m = np.einsum("ij,jk,ik->i", Z, self._cov_inv, Z)
            return np.sqrt(np.maximum(m, 0.0))
        out = np.empty(Z.shape[0], dtype=float)
        for i, z in enumerate(Z):
            d = np.sqrt(((Z - z) ** 2).sum(axis=1))
            d.sort()
            out[i] = d[1 : self.k + 1].mean()  # skip the zero self-distance
        return out


# --------------------------------------------------------------------------- #
# Split-conformal calibration of the ensemble sigma
# --------------------------------------------------------------------------- #
@dataclass
class ConformalCalibrator:
    """Split-conformal scaling so predicted intervals hit nominal coverage.

    Nonconformity score is the *normalized* residual s_i = |y_i - ŷ_i| / σ_i on a
    held-out set. For a target central coverage ``p`` (e.g. 0.68), the calibrated
    multiplier is the finite-sample ``p``-quantile of {s_i}; a calibrated σ is
    ``σ * q_p``. Because the score is normalized, this both rescales the *overall*
    dispersion (fixing the ~2x under-dispersion) and preserves the σ *ranking*
    the ensemble already provides.
    """

    coverage: float = 0.68
    q: float | None = None  # fitted multiplier

    def fit(self, residuals: np.ndarray, sigmas: np.ndarray) -> "ConformalCalibrator":
        r = np.abs(np.asarray(residuals, dtype=float))
        s = np.asarray(sigmas, dtype=float)
        mask = np.isfinite(r) & np.isfinite(s) & (s > 0)
        scores = r[mask] / s[mask]
        n = scores.size
        if n == 0:
            raise ValueError("no valid (residual, sigma) pairs to calibrate on")
        # Finite-sample conformal level: ceil((n+1)*p)/n, clipped to [0, 1].
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.q = float(np.quantile(scores, level, method="higher"))
        return self

    def calibrate(self, sigmas: np.ndarray) -> np.ndarray:
        """Return conformally-scaled σ (``σ * q``)."""
        if self.q is None:
            raise RuntimeError("ConformalCalibrator.calibrate called before fit")
        return np.asarray(sigmas, dtype=float) * self.q


def empirical_coverage(
    residuals: np.ndarray, sigmas: np.ndarray, n_sigma: float = 1.0
) -> float:
    """Fraction of points whose |residual| ≤ ``n_sigma`` · σ."""
    r = np.abs(np.asarray(residuals, dtype=float))
    s = np.asarray(sigmas, dtype=float)
    mask = np.isfinite(r) & np.isfinite(s) & (s > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(r[mask] <= n_sigma * s[mask]))


# --------------------------------------------------------------------------- #
# Combined OOD-aware predictive uncertainty
# --------------------------------------------------------------------------- #
def predictive_uncertainty(
    sigma_cal: np.ndarray,
    ood_norm: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Inflate the calibrated σ by the (normalized) OOD distance.

        σ_total = σ_cal * (1 + alpha * ood_norm)

    ``ood_norm`` in [0, 1] is the training-referenced OOD percentile
    (:meth:`OODDetector.percentile`). In-distribution points (ood_norm≈0) keep
    ``σ_cal``; a maximally-novel point (ood_norm≈1) is inflated by ``1+alpha``.
    ``alpha`` controls how aggressively novelty widens the interval (default 1.0
    ⇒ up to a 2x widening at the training frontier, unbounded only if a query is
    scored beyond it, where the percentile saturates at 1.0).
    """
    sc = np.asarray(sigma_cal, dtype=float)
    on = np.clip(np.asarray(ood_norm, dtype=float), 0.0, None)
    return sc * (1.0 + alpha * on)
