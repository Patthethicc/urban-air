def test_patch_and_window_match_same_station_and_time():
    """Pick one row i. By hand, check that X_spatial[i] equals the raster patch for
    meta.station_id[i], and X_temporal[i] equals that station's raw rows ending at
    meta.timestamp[i]. Use a tiny synthetic raster/dataframe so you know the answer."""
    # TODO
    pass


def test_split_is_chronological_and_leak_free():
    # TODO: max(train ts) < min(val ts) < min(test ts); no window straddles a boundary
    pass


def test_scalers_fit_on_train_only():
    # TODO: scaler mean equals train mean, not full-data mean
    pass
