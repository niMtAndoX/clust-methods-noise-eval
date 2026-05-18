import numpy as np
from sklearn.metrics import pairwise_distances


def _regular_cluster_labels(labels, noise_label):
    """Return all labels except the noise label."""
    unique_labels = np.unique(labels)

    if noise_label is None:
        return list(unique_labels)

    return [label for label in unique_labels if label != noise_label]


def modified_silhouette_score(
    X,
    labels,
    noise_label=-1,
    metric="euclidean",
    return_samples=False,
    **metric_kwargs
):
    """
    Modified silhouette coefficient for clustering results with noise.

    For regular clustered points x_i in X_clust:

        s_i^+ = (b_i - a_i) / max(a_i, b_i)

    For noise points x_i in X_noise:

        s_i^+ = min(1, 1 - (c_i - b_i) / max(b_i, c_i))

    Definitions:
    ------------
    a_i:
        Mean distance from x_i to all other points in its own cluster.

    b_i:
        For regular cluster points:
            Minimum mean distance from x_i to another regular cluster.
        For noise points:
            Minimum mean distance from x_i to a regular cluster.

    c_i:
        Mean distance from a noise point x_i to all other noise points.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.

    labels : array-like, shape (n_samples,)
        Predicted cluster labels.

    noise_label : int, str, or None, default=-1
        Label used for noise points.
        DBSCAN usually uses -1.
        RTKM may use k as the noise label.
        For algorithms without explicit noise labels, use noise_label=None.

    metric : str, default="euclidean"
        Distance metric passed to sklearn.metrics.pairwise_distances.

    return_samples : bool, default=False
        If True, return the score for each sample as well.

    Returns:
    --------
    float
        Mean modified silhouette score.
        Higher is better.

    np.ndarray, optional
        Per-sample scores if return_samples=True.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X and labels must contain the same number of samples. "
            f"Got {X.shape[0]} samples and {labels.shape[0]} labels."
        )

    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError("At least two samples are required.")

    regular_labels = _regular_cluster_labels(labels, noise_label)

    has_noise = noise_label is not None and np.any(labels == noise_label)

    if len(regular_labels) < 2 and not has_noise:
        result = np.nan
        if return_samples:
            return result, np.full(n_samples, np.nan)
        return result

    distances = pairwise_distances(X, metric=metric, **metric_kwargs)

    sample_scores = np.zeros(n_samples, dtype=float)

    noise_mask = (
        np.zeros(n_samples, dtype=bool)
        if noise_label is None
        else labels == noise_label
    )

    for i in range(n_samples):
        current_label = labels[i]

        # Case 1: regular clustered point
        if noise_label is None or current_label != noise_label:
            same_cluster_mask = labels == current_label
            same_cluster_mask[i] = False

            if np.any(same_cluster_mask):
                a_i = float(np.mean(distances[i, same_cluster_mask]))
            else:
                a_i = 0.0

            b_values = []

            for other_label in regular_labels:
                if other_label == current_label:
                    continue

                other_cluster_mask = labels == other_label

                if np.any(other_cluster_mask):
                    b_values.append(float(np.mean(distances[i, other_cluster_mask])))

            if len(b_values) == 0:
                sample_scores[i] = 0.0
                continue

            b_i = min(b_values)
            denominator = max(a_i, b_i)

            if denominator == 0:
                sample_scores[i] = 0.0
            else:
                sample_scores[i] = (b_i - a_i) / denominator

        # Case 2: noise point
        else:
            b_values = []

            for cluster_label in regular_labels:
                cluster_mask = labels == cluster_label

                if np.any(cluster_mask):
                    b_values.append(float(np.mean(distances[i, cluster_mask])))

            if len(b_values) == 0:
                sample_scores[i] = 0.0
                continue

            b_i = min(b_values)

            other_noise_mask = noise_mask.copy()
            other_noise_mask[i] = False

            if np.any(other_noise_mask):
                c_i = float(np.mean(distances[i, other_noise_mask]))
            else:
                c_i = 0.0

            denominator = max(b_i, c_i)

            if denominator == 0:
                sample_scores[i] = 0.0
            else:
                sample_scores[i] = min(
                    1.0,
                    1.0 - ((c_i - b_i) / denominator)
                )

    score = float(np.mean(sample_scores))

    if return_samples:
        return score, sample_scores

    return score


def modified_davies_bouldin_score(
    X,
    labels,
    noise_label=-1,
    metric="euclidean",
    return_cluster_scores=False,
    **metric_kwargs
):
    """
    Modified Davies-Bouldin index for clustering results with noise.

    For regular clusters C_i in pi_X_clust:

        DB_i^+ = max_{i != j} { (a_i + a_j) / delta_ij }

    For the noise cluster C_i = X_noise:

        DB_noise^+ = max_n { D_NR / min_j delta_nj }

    Definitions:
    ------------
    a_i:
        Mean distance of points in cluster C_i to the centroid of C_i.

    delta_ij:
        Distance between centroids of regular clusters C_i and C_j.

    D_NR:
        Mean pairwise distance between noise points.

    delta_nj:
        Distance from noise point x_n to centroid of regular cluster C_j.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.

    labels : array-like, shape (n_samples,)
        Predicted cluster labels.

    noise_label : int, str, or None, default=-1
        Label used for noise points.
        DBSCAN usually uses -1.
        RTKM may use k as the noise label.
        For algorithms without explicit noise labels, use noise_label=None.

    metric : str, default="euclidean"
        Distance metric passed to sklearn.metrics.pairwise_distances.

    return_cluster_scores : bool, default=False
        If True, return DB_i^+ for each cluster.

    Returns:
    --------
    float
        Modified Davies-Bouldin index.
        Lower is better.

    dict, optional
        Per-cluster scores if return_cluster_scores=True.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")

    if X.shape[0] != labels.shape[0]:
        raise ValueError(
            f"X and labels must contain the same number of samples. "
            f"Got {X.shape[0]} samples and {labels.shape[0]} labels."
        )

    regular_labels = _regular_cluster_labels(labels, noise_label)

    if len(regular_labels) == 0:
        result = np.nan
        if return_cluster_scores:
            return result, {}
        return result

    noise_mask = (
        np.zeros(X.shape[0], dtype=bool)
        if noise_label is None
        else labels == noise_label
    )

    has_noise = np.any(noise_mask)

    centers = {}
    scatters = {}

    for label in regular_labels:
        cluster_points = X[labels == label]

        if len(cluster_points) == 0:
            continue

        center = np.mean(cluster_points, axis=0)
        centers[label] = center

        distances_to_center = pairwise_distances(
            cluster_points,
            center.reshape(1, -1),
            metric=metric,
            **metric_kwargs
        )

        scatters[label] = float(np.mean(distances_to_center))

    cluster_scores = {}

    for label_i in regular_labels:
        ratios = []

        for label_j in regular_labels:
            if label_i == label_j:
                continue

            center_i = centers[label_i].reshape(1, -1)
            center_j = centers[label_j].reshape(1, -1)

            delta_ij = pairwise_distances(
                center_i,
                center_j,
                metric=metric,
                **metric_kwargs
            )[0, 0]

            if delta_ij == 0:
                ratios.append(np.inf)
            else:
                ratios.append((scatters[label_i] + scatters[label_j]) / delta_ij)

        if len(ratios) == 0:
            cluster_scores[label_i] = np.nan
        else:
            cluster_scores[label_i] = float(np.max(ratios))

    if has_noise:
        noise_points = X[noise_mask]

        if len(noise_points) <= 1:
            D_NR = 0.0
        else:
            noise_distances = pairwise_distances(
                noise_points,
                metric=metric,
                **metric_kwargs
            )

            upper_triangle = np.triu_indices_from(noise_distances, k=1)
            D_NR = float(np.mean(noise_distances[upper_triangle]))

        center_matrix = np.vstack([centers[label] for label in regular_labels])

        noise_to_centers = pairwise_distances(
            noise_points,
            center_matrix,
            metric=metric,
            **metric_kwargs
        )

        nearest_cluster_distances = np.min(noise_to_centers, axis=1)

        noise_ratios = []

        for distance_to_nearest_cluster in nearest_cluster_distances:
            if distance_to_nearest_cluster == 0:
                if D_NR == 0:
                    noise_ratios.append(0.0)
                else:
                    noise_ratios.append(np.inf)
            else:
                noise_ratios.append(D_NR / distance_to_nearest_cluster)

        cluster_scores[noise_label] = float(np.max(noise_ratios))

    values = np.array(list(cluster_scores.values()), dtype=float)

    if np.all(np.isnan(values)):
        score = np.nan
    else:
        score = float(np.nanmean(values))

    if return_cluster_scores:
        return score, cluster_scores

    return score
