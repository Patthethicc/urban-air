# Dataset comparison on out-of-fold predictions (3 runs, 1 arms)

Scored on the 786,682 (station, hour) rows that every run has and agrees on, 54 training stations. Out-of-fold only: the test partition is not used.

| Arm | Runs | RMSE | RMSE sd | MAE | R2 | Bias | Gap to best | 95% CI of gap | Tied | Common rows / own |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| from_2023_with_preinstall | 3 | 6.517 | 0.005 | 3.701 | 0.696 | -0.16 | best |  |  | 100% |

**Best by RMSE: `from_2023_with_preinstall`.** Tied with it: none. With the preference order ['full', 'from_2023', 'from_2024', 'full_with_preinstall', 'from_2023_with_preinstall', 'from_2024_with_preinstall', 'raw'], the choice is **`from_2023_with_preinstall`**.

A gap is real only when its CI excludes 0 and it exceeds the seed-to-seed spread (RMSE sd); with one run per arm there is no spread to compare.

## Checks

- Same CV folds for every station in every run: **yes** (assignments used: by_station)
- Rows dropped because runs disagree on the true value: 0
- Largest station is 3.3% of the common rows, the five largest 16.7%; smallest has 3,186 rows
- Bootstrap: 2000 resamples of whole stations (seed 42)
