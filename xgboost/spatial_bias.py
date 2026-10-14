import numpy as np
import pandas as pd

# ADDITIONAL NOTES FOR SPATIAL_BIAS
# implements Section 4.4.3 Spatial Bias Detection from the thesis:
#   1) aggregate out-of-fold residuals to one mean value per Grid_ID
#   2) build a K-Nearest-Neighbors spatial weights matrix over grid centroids
#   3) Global Moran's I to test for overall spatial autocorrelation of residuals
#   4) Local Moran's I (LISA) to localize significant clusters when global I is significant
# implemented directly with numpy so the pipeline doesn't need libpysal/esda.


def aggregate_residuals_by_grid(oof_df, mapping_csv_path='sensor_to_grid_map.csv'):
    """Mean residual per Grid_ID, joined back to projected centroid coordinates."""
    df_map = pd.read_csv(mapping_csv_path)
    df_map = df_map.rename(columns={'sensor_id': 'Location', 'id': 'Grid_ID'})
    df_map = df_map.drop_duplicates(subset='Location', keep='first')
    df_map['cx'] = (df_map['left'] + df_map['right']) / 2
    df_map['cy'] = (df_map['top'] + df_map['bottom']) / 2
    centroids = df_map.groupby('Grid_ID')[['cx', 'cy']].first().reset_index()

    grid_residuals = (
        oof_df.groupby('Grid_ID')['residual']
        .mean()
        .reset_index()
        .merge(centroids, on='Grid_ID', how='left')
    )
    return grid_residuals


def build_knn_weights(grid_residuals, k=4):
    """Row-standardized K-Nearest-Neighbors spatial weights matrix (n x n)."""
    coords = grid_residuals[['cx', 'cy']].values
    n = len(coords)
    k = min(k, n - 1)  # guard against small pilot samples

    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)

    W = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dists[i])[:k]
        W[i, nearest] = 1.0

    # row-standardize so each row sums to 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    return W


def global_morans_i(values, W, n_permutations=999, random_state=42):
    """Global Moran's I with a permutation-based pseudo p-value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    z = values - values.mean()
    S0 = W.sum()

    def compute_I(zv):
        num = zv @ W @ zv
        den = (zv ** 2).sum()
        return (n / S0) * (num / den) if den > 0 and S0 > 0 else 0.0

    I_obs = compute_I(z)

    rng = np.random.default_rng(random_state)
    perm_Is = np.empty(n_permutations)
    for p in range(n_permutations):
        z_perm = rng.permutation(z)
        perm_Is[p] = compute_I(z_perm)

    p_value = (np.sum(np.abs(perm_Is) >= np.abs(I_obs)) + 1) / (n_permutations + 1)
    return I_obs, p_value


def local_morans_i(values, W, n_permutations=999, random_state=42):
    """Anselin's Local Moran's I (LISA) per observation, with pseudo p-values
    and High-High / Low-Low / High-Low / Low-High quadrant labels."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    z = values - values.mean()
    m2 = (z ** 2).sum() / n

    lag = W @ z
    I_local = (z / m2) * lag

    rng = np.random.default_rng(random_state)
    p_sim = np.empty(n)
    for i in range(n):
        others = np.delete(z, i)
        perm_lags = np.array([
            (W[i] @ np.insert(rng.permutation(others), i, z[i]))
            for _ in range(n_permutations)
        ])
        perm_I = (z[i] / m2) * perm_lags
        p_sim[i] = (np.sum(np.abs(perm_I) >= np.abs(I_local[i])) + 1) / (n_permutations + 1)

    quadrant = []
    for zi, li in zip(z, lag):
        if zi >= 0 and li >= 0:
            quadrant.append('High-High')
        elif zi < 0 and li < 0:
            quadrant.append('Low-Low')
        elif zi >= 0 and li < 0:
            quadrant.append('High-Low')
        else:
            quadrant.append('Low-High')

    return pd.DataFrame({'local_I': I_local, 'p_sim': p_sim, 'quadrant': quadrant})
