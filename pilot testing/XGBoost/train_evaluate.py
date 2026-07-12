import numpy as np
import pandas as pd
import xgboost as xgb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from prep_n_proc import df_clean, X, y
from spatial_cv import get_outer_folds, get_inner_folds
from spatial_bias import build_knn_weights, global_morans_i, local_morans_i, aggregate_residuals_by_grid

# needs prep_n_proc.py for data, spatial_cv.py for fold assignment helpers

# Outer spatial fold assignment (k=4) using nested cv
outer_fold_map = get_outer_folds('sensor_to_grid_map.csv', k=4, random_state=42)
df_clean['outer_fold'] = df_clean['Grid_ID'].map(outer_fold_map) # type: ignore[arg-type]

print("Outer fold sizes (rows):")
print(df_clean['outer_fold'].value_counts().sort_index())

N_TRIALS = 20  # can be changed, but few for now since smaller dataset

SEARCH_SPACE = [
    Integer(2, 5, name='max_depth'),
    Integer(1, 15, name='min_child_weight'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
    Real(0.0, 5.0, name='gamma'),
    Real(0.1, 10.0, prior='log-uniform', name='reg_lambda'),
    Real(0.0, 10.0, name='reg_alpha'),
]

fold_results = []
oof_records = []

for outer_f in sorted(df_clean['outer_fold'].unique()):
    print(f"\n--- Outer fold {outer_f} ---")

    train_mask = (df_clean['outer_fold'] != outer_f).values
    test_mask = (df_clean['outer_fold'] == outer_f).values

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # inner folds are built only on this outer fold's training cells
    held_out_grid_ids = df_clean.loc[df_clean['outer_fold'] == outer_f, 'Grid_ID'].unique()
    inner_fold_map = get_inner_folds('sensor_to_grid_map.csv',
                                      exclude_grid_ids=held_out_grid_ids,
                                      k=3, random_state=42)
    inner_fold_labels = df_clean.loc[train_mask, 'Grid_ID'].map(inner_fold_map).values
    call_records = []  # tracks avg_best_iteration per gp_minimize call

    @use_named_args(SEARCH_SPACE)
    def objective(**params):
        inner_rmses, best_iters = [], []
        for inner_f in sorted(set(inner_fold_labels)):
            itr_mask = inner_fold_labels != inner_f
            ite_mask = inner_fold_labels == inner_f

            model = xgb.XGBRegressor(
                n_estimators=500, early_stopping_rounds=20,
                random_state=42, **params,
            )
            model.fit(X_train[itr_mask], y_train[itr_mask],
                      eval_set=[(X_train[ite_mask], y_train[ite_mask])], verbose=False)

            preds = model.predict(X_train[ite_mask])
            inner_rmses.append(sqrt(mean_squared_error(y_train[ite_mask], preds)))
            best_iters.append(model.best_iteration)

        avg_rmse = float(np.mean(inner_rmses))
        call_records.append({'params': params, 'avg_rmse': avg_rmse,
                              'avg_best_iteration': float(np.mean(best_iters))})
        return avg_rmse

    result = gp_minimize(objective, SEARCH_SPACE, n_calls=N_TRIALS,
                          n_initial_points=8, random_state=42, verbose=False)

    best_call = min(call_records, key=lambda r: r['avg_rmse'])
    best_params = best_call['params']
    best_n_estimators = max(10, round(best_call['avg_best_iteration']))
    print(f"Best inner-CV RMSE: {best_call['avg_rmse']:.3f} | n_estimators={best_n_estimators}")
    print(f"Best params: {best_params}")

    # fit on the full outer-train set
    # (all inner folds combined) using the tuned hyperparameters
    final_model = xgb.XGBRegressor(n_estimators=best_n_estimators, random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    preds_test = final_model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    r2 = r2_score(y_test, preds_test)
    print(f"Outer-test  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  (n={test_mask.sum()})")

    fold_results.append({
        'outer_fold': outer_f, 'n_test_rows': int(test_mask.sum()),
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'best_params': best_params, 'n_estimators': best_n_estimators,
    })

    oof_chunk = df_clean.loc[test_mask, ['Location', 'Grid_ID', 'latitude', 'longitude', 'DateTime', 'Value']].copy()
    oof_chunk['predicted'] = preds_test
    oof_chunk['residual'] = oof_chunk['predicted'] - oof_chunk['Value']
    oof_chunk['outer_fold'] = outer_f
    oof_records.append(oof_chunk)

results_df = pd.DataFrame(fold_results)
oof_df = pd.concat(oof_records, ignore_index=True)

print("\n=== Nested spatial CV summary ===")
print(results_df[['outer_fold', 'n_test_rows', 'rmse', 'mae', 'r2']])
print(f"\nMean RMSE: {results_df['rmse'].mean():.3f} (+/- {results_df['rmse'].std():.3f})")
print(f"Mean MAE:  {results_df['mae'].mean():.3f} (+/- {results_df['mae'].std():.3f})")
print(f"Mean R2:   {results_df['r2'].mean():.3f} (+/- {results_df['r2'].std():.3f})")

# SPATIAL BIAS DETECTION
grid_residuals = aggregate_residuals_by_grid(oof_df, 'sensor_to_grid_map.csv')
print(f"\nAggregated to {len(grid_residuals)} grid cells for Moran's I (mean residual per cell):")
print(grid_residuals[['Grid_ID', 'residual']].sort_values(by=['residual']))  # type: ignore[call-overload]

w = build_knn_weights(grid_residuals, k=4)
I, p_value = global_morans_i(grid_residuals['residual'].values, w)

print(f"\nGlobal Moran's I: {I:.3f}  (p = {p_value:.3f})")
if p_value < 0.05:
    direction = "positive (residuals of similar sign/magnitude cluster together)" if I > 0 \
        else "negative (dispersed/checkerboard pattern)"
    print(f"-> Statistically significant spatial clustering detected: {direction}")
    print("-> Running Local Moran's I (LISA) to find WHERE...")

    lisa_df = local_morans_i(grid_residuals['residual'].values, w)
    lisa_df = pd.concat([grid_residuals[['Grid_ID', 'residual']].reset_index(drop=True), lisa_df], axis=1)
    print(lisa_df.sort_values(by=['p_sim']))  # type: ignore[call-overload]
else:
    print("-> No statistically significant spatial pattern in residuals "
          "(errors appear geographically random at this alpha).")