"""Paper-faithful Robust Trimmed k-Means (RTKM).

This implementation follows Algorithm 1 from Dorabiala, Kutz & Aravkin
"Robust trimmed k-means" (Pattern Recognition Letters, 2022) as closely as
possible while keeping the interface of the originally uploaded RTKM.py.

Expected data layout: data has shape (m, N), where m is the number of
features/dimensions and N is the number of data points.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def _round_half_up(x: float) -> int:
    """Nearest integer, with half-integers rounded upward."""
    return int(np.floor(x + 0.5))


def project_capped_simplex_rows(
    z: np.ndarray,
    target_sum: float,
    lower: float = 0.0,
    upper: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """Project each row of z onto a capped simplex.

    Solves, row-wise,

        argmin_x ||x - z||_2^2
        subject to lower <= x_i <= upper and sum_i x_i = target_sum.

    The RTKM paper constrains both W-columns and v to capped simplexes:
    W[:, i] in Delta_s and v in Delta_{N - [alpha N]}.

    Parameters
    ----------
    z:
        1D or 2D array to project. If 1D, it is treated as one row.
    target_sum:
        Desired row sum.
    lower, upper:
        Component-wise bounds.
    max_iter:
        Maximum number of bisection iterations.
    tol:
        Tolerance for the bisection interval.

    Returns
    -------
    np.ndarray
        Projected copy with the same dimensionality as the input.
    """
    arr = np.asarray(z, dtype=float)
    was_1d = arr.ndim == 1
    rows = arr.reshape(1, -1).copy() if was_1d else arr.copy()

    n = rows.shape[1]
    if not lower * n <= target_sum <= upper * n:
        raise ValueError(
            "Target sum is infeasible for the given bounds: "
            f"target_sum={target_sum}, bounds=[{lower}, {upper}], n={n}."
        )

    for row in rows:
        # The solution has the form clip(row + lambda, lower, upper).
        lo = np.min(lower - row)
        hi = np.max(upper - row)

        for _ in range(max_iter):
            lam = 0.5 * (lo + hi)
            projected = np.clip(row + lam, lower, upper)
            row_sum = projected.sum()

            if row_sum > target_sum:
                hi = lam
            else:
                lo = lam

            if hi - lo <= tol:
                break

        row[:] = np.clip(row + 0.5 * (lo + hi), lower, upper)

    return rows.ravel() if was_1d else rows


# Compatibility aliases for code that imported the original helper names.
def SimplexProx(z: np.ndarray, a: float) -> np.ndarray:
    """Project rows onto the nonnegative simplex with row sum a.

    This matches the helper name from the uploaded implementation. The paper
    itself uses a capped simplex for W; the RTKM class below therefore calls
    project_capped_simplex_rows directly.
    """
    arr = np.asarray(z, dtype=float)
    was_1d = arr.ndim == 1
    rows = arr.reshape(1, -1).copy() if was_1d else arr.copy()

    if a < 0:
        raise ValueError("simplex target sum must be nonnegative.")

    for row in rows:
        # Projection onto {x >= 0, sum(x) = a}.
        u = np.sort(row)[::-1]
        cssv = np.cumsum(u) - a
        ind = np.arange(1, row.size + 1)
        cond = u - cssv / ind > 0
        if not np.any(cond):
            theta = 0.0
        else:
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / rho
        row[:] = np.maximum(row - theta, 0.0)

    return rows.ravel() if was_1d else rows


def proj_csimplex(
    z: np.ndarray,
    a: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """In-place-compatible wrapper for capped-simplex projection."""
    projected = project_capped_simplex_rows(z, a, lower=lower, upper=upper)
    z[...] = projected
    return z


class RTKM:
    """Robust Trimmed k-Means following Algorithm 1 of the RTKM paper.

    The method alternates between updates of

    * cluster centers C,
    * membership weights W, and
    * inlier/outlier weights v.

    For single-membership clustering, set num_members=1. After convergence,
    the implementation can harden W with argmax, as described in Algorithm 1.
    """

    def __init__(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array with shape (m, N).")

        self.data = data
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.continuous_weights: Optional[np.ndarray] = None
        self.outliers: Optional[np.ndarray] = None
        self.obj_hist: Optional[list[float]] = None
        self.err_hist: Optional[list[float]] = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _squared_distances(self, centers: np.ndarray) -> np.ndarray:
        """Return D[j, i] = ||x_i - c_j||^2."""
        data_norm = np.sum(self.data ** 2, axis=0)          # (N,)
        center_norm = np.sum(centers ** 2, axis=0)          # (k,)
        distances = center_norm[:, None] - 2.0 * centers.T @ self.data + data_norm[None, :]
        # Numerical roundoff can create tiny negative values.
        return np.maximum(distances, 0.0)

    def _initialize_weights(
        self,
        k: int,
        n: int,
        num_members: float,
        rng: np.random.Generator,
        init_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        if init_weights is not None:
            weights = np.asarray(init_weights, dtype=float)
            if weights.shape != (k, n):
                raise ValueError(f"init_weights must have shape {(k, n)}, got {weights.shape}.")
            # Ensure feasibility.
            return project_capped_simplex_rows(weights.T, num_members, 0.0, 1.0).T

        # The paper's relaxed formulation works with continuous W-values.
        # Random initialization avoids the zero-denominator problem in the first
        # center update and matches the paper's description around Fig. 2.
        weights = rng.random((n, k))
        weights = project_capped_simplex_rows(weights, num_members, 0.0, 1.0)
        return weights.T

    def perform_clustering(
        self,
        k: int,
        percent_outliers: float,
        tol: float = 1e-6,
        max_iter: int = 100,
        init_centers: Optional[np.ndarray] = None,
        init_weights: Optional[np.ndarray] = None,
        num_members: float = 1.0,
        random_state: Optional[int] = None,
        harden_single_membership: bool = True,
        verbose: bool = False,
    ) -> None:
        """Run RTKM.

        Parameters
        ----------
        k:
            Number of clusters.
        percent_outliers:
            Expected outlier proportion alpha. The number of inliers is
            h = N - [alpha N], where [.] means nearest integer with half values
            rounded upward, as in the paper.
        tol:
            Convergence tolerance based on changes in C, W and v.
        max_iter:
            Maximum number of iterations.
        init_centers:
            Optional initial centers of shape (m, k).
        init_weights:
            Optional initial membership matrix of shape (k, N).
        num_members:
            Parameter s. Each point has membership weights summing to s, with
            every weight constrained to [0, 1].
        random_state:
            Seed for reproducibility.
        harden_single_membership:
            If True and num_members == 1, replace continuous W by one-hot
            argmax assignments after convergence, as described in Algorithm 1.
            The continuous weights remain available as self.continuous_weights.
        verbose:
            Print progress information.
        """
        if not 0.0 <= percent_outliers <= 1.0:
            raise ValueError("percent_outliers must be in [0, 1].")
        if not 1 <= k:
            raise ValueError("k must be at least 1.")
        if not 0.0 <= num_members <= k:
            raise ValueError("num_members must satisfy 0 <= num_members <= k.")

        m, n = self.data.shape
        if k > n:
            raise ValueError("k must not exceed the number of data points.")

        rng = np.random.default_rng(random_state)

        # Paper: v in Delta_{N - [alpha N]}, where [.] is nearest integer with
        # half-integers rounded upward.
        expected_outliers = _round_half_up(percent_outliers * n)
        h = n - expected_outliers

        if init_centers is None:
            # Choose distinct data points as centers. The paper allows any
            # initialization scheme; using replace=False avoids duplicate centers.
            center_idx = rng.choice(n, size=k, replace=False)
            centers = self.data[:, center_idx].copy()
        else:
            centers = np.asarray(init_centers, dtype=float).copy()
            if centers.shape != (m, k):
                raise ValueError(f"init_centers must have shape {(m, k)}, got {centers.shape}.")

        weights = self._initialize_weights(k, n, num_members, rng, init_weights)

        # Feasible initial v. If alpha=0, this is exactly all ones.
        inlier_weights = project_capped_simplex_rows(np.ones(n), h, 0.0, 1.0)

        dk = 1.1
        ek = 1.1

        obj_hist: list[float] = []
        err_hist: list[float] = []

        err = np.inf
        converged = False
        eps = np.finfo(float).eps

        for iteration in range(1, max_iter + 1):
            centers_old = centers.copy()
            weights_old = weights.copy()
            inlier_weights_old = inlier_weights.copy()

            # Algorithm 1, line 4:
            # c_j = sum_i v_i w_{j,i} x_i / sum_i v_i w_{j,i}
            weighted_memberships = weights * inlier_weights[None, :]  # (k, N)
            denominators = weighted_memberships.sum(axis=1)           # (k,)

            centers_new = centers.copy()
            nonempty = denominators > eps
            centers_new[:, nonempty] = (
                self.data @ weighted_memberships[nonempty, :].T
            ) / denominators[nonempty][None, :]
            # If a cluster is numerically empty, keep its previous center. The
            # paper does not specify this edge case; this avoids NaNs.

            distances = self._squared_distances(centers_new)          # (k, N)

            # Algorithm 1, line 5:
            # W[:, i] = proj_Delta_s(W[:, i] - (1/dk) v_i * D[:, i])
            weights_candidate = (
                weights.T - (1.0 / dk) * inlier_weights[:, None] * distances.T
            )
            weights_new = project_capped_simplex_rows(
                weights_candidate, num_members, lower=0.0, upper=1.0
            ).T

            # Algorithm 1, line 6:
            # v = proj_Delta_{N-[alpha N]}(v - (1/ek) * sum_j W[j, :] * D[j, :])
            weighted_distances = np.sum(weights_new * distances, axis=0)  # (N,)
            inlier_candidate = inlier_weights - (1.0 / ek) * weighted_distances
            inlier_weights_new = project_capped_simplex_rows(
                inlier_candidate, h, lower=0.0, upper=1.0
            )

            centers = centers_new
            weights = weights_new
            inlier_weights = inlier_weights_new

            centers_err = np.linalg.norm(centers - centers_old)
            weights_err = np.linalg.norm(weights - weights_old)
            outliers_err = np.linalg.norm(inlier_weights - inlier_weights_old)
            err = centers_err + dk * weights_err + ek * outliers_err

            # Objective from Eq. (5): sum_i v_i * sum_j w_{j,i} ||x_i-c_j||^2
            distances_for_obj = self._squared_distances(centers)
            obj = float(np.sum(inlier_weights * np.sum(weights * distances_for_obj, axis=0)))
            obj_hist.append(obj)
            err_hist.append(float(err))

            if verbose and (iteration == 1 or iteration % 100 == 0):
                print(f"Iteration {iteration}: objective={obj:.6g}, err={err:.6g}")

            if err < tol:
                converged = True
                break

        if verbose and not converged:
            print("RTKM reached maximum number of iterations")

        self.centers = centers
        self.continuous_weights = weights.copy()

        if harden_single_membership and np.isclose(num_members, 1.0):
            hard_weights = np.zeros_like(weights)
            hard_weights[np.argmax(weights, axis=0), np.arange(n)] = 1.0
            self.weights = hard_weights
        else:
            self.weights = np.maximum(weights, 0.0)

        # Keep continuous inlier weights. Do not cast to int, because the paper's
        # relaxation permits values in [0, 1].
        self.outliers = inlier_weights
        self.obj_hist = obj_hist
        self.err_hist = err_hist
        self.n_iter_ = len(obj_hist)
        self.converged_ = converged

    def return_clusters(self, outlier_tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """Return predicted cluster labels and detected outlier indices.

        Outliers receive label k, matching the behavior of the uploaded code.
        A point is treated as an outlier when its inlier weight v_i is close to 0.
        """
        if self.weights is None or self.outliers is None:
            raise RuntimeError("Call perform_clustering before return_clusters.")

        pred_clusters = np.argmax(self.weights, axis=0).astype(int)
        pred_outliers = np.where(self.outliers <= outlier_tol)[0]
        pred_clusters[pred_outliers] = self.weights.shape[0]
        return pred_clusters, pred_outliers

    def performance_report(self) -> None:
        """Plot objective value and optimality condition over iterations."""
        if self.obj_hist is None or self.err_hist is None:
            raise RuntimeError("Call perform_clustering before performance_report.")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(self.obj_hist)
        ax[0].set_title("function value")
        ax[1].semilogy(self.err_hist)
        ax[1].set_title("optimality condition")
        fig.suptitle("RTKM Performance Report")
