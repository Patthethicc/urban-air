import pandas as pd
from sklearn.cluster import KMeans

# ADDITIONAL NOTES FOR SPATIAL_CV
# runs k-means on grid cell centroids to assign spatial folds
# used for both outer and inner cv loops

# gets one row per unique Grid_ID with its projected centroid (cx, cy)
# in EPSG:32651 (meters), suitable for k-means distance-based clustering
def get_grid_centroids(mapping_csv_path='sensor_to_grid_map.csv'):
    df = pd.read_csv(mapping_csv_path)
    df = df.rename(columns={'sensor_id': 'Location', 'id': 'Grid_ID'})
    df = df.drop_duplicates(subset='Location', keep='first')
    df['cx'] = (df['left'] + df['right']) / 2
    df['cy'] = (df['top'] + df['bottom']) / 2
    centroids = df.groupby('Grid_ID')[['cx', 'cy']].first().reset_index()
    return centroids

# assigns spatial folds to grid cell centroids using k-means clustering
def assign_spatial_folds(centroids, k, random_state=42):
    centroids = centroids.copy()
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    centroids['fold'] = km.fit_predict(centroids[['cx', 'cy']])
    return centroids

# assigns outer folds to grid cell centroids
def get_outer_folds(mapping_csv_path='sensor_to_grid_map.csv', k=4, random_state=42):
    centroids = get_grid_centroids(mapping_csv_path)
    centroids = assign_spatial_folds(centroids, k=k, random_state=random_state)
    return dict(zip(centroids['Grid_ID'], centroids['fold']))

# assigns inner folds to grid cell centroids
def get_inner_folds(mapping_csv_path='sensor_to_grid_map.csv', exclude_grid_ids=None,
                     k=3, random_state=42):
    centroids = get_grid_centroids(mapping_csv_path)
    if exclude_grid_ids is not None:
        centroids = centroids[~centroids['Grid_ID'].isin(exclude_grid_ids)].reset_index(drop=True)
    centroids = assign_spatial_folds(centroids, k=k, random_state=random_state)
    return dict(zip(centroids['Grid_ID'], centroids['fold']))


if __name__ == '__main__':
    centroids = get_grid_centroids()
    outer = assign_spatial_folds(centroids, k=4)
    print("Outer fold sizes (cells):", outer['fold'].value_counts().sort_index().to_dict())

    for f in sorted(outer['fold'].unique()):
        held_out_ids = outer.loc[outer['fold'] == f, 'Grid_ID'].tolist()
        inner = get_inner_folds(exclude_grid_ids=held_out_ids, k=3)
        inner_sizes = pd.Series(list(inner.values())).value_counts().sort_index().to_dict()
        print(f"Outer fold {f} held out ({len(held_out_ids)} cells) -> "
              f"inner k=3 sizes: {inner_sizes}")
