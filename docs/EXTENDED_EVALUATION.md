# Extended evaluation and dataset expansion

## Data and scope

The Delhi source now contains 966,067 hourly pollutant observations: 317,627 PM2.5, 323,061 PM10 and 325,379 NO2 rows from 39 catalog entries in 2024. This adds 648,440 observations, a 204.2% increase over the original Delhi source. The PM2.5 experiment retains the original 36 distinct stations and 300,355 valid labels. Additional pollutant rows support context exploration, not extra PM2.5 training labels. Beijing contributes 420,768 station-hour source rows across 12 stations. These counts exclude duplicate ZIP representations, weekly provenance copies and synthetic derivatives.

## Spatial cross-validation

Five deterministic station folds rotate through each geographic client. Every retained location is unseen exactly once per seed. Four folds hold out eight locations and the fifth holds out four. Donors, training labels, validation labels and calibration exclude all stations in the current holdout. Future-time test boundaries are unchanged. Three seeds (17, 29, 43), 20 rounds, 12 local steps and Adam 0.002 are used throughout. Summary values below are unweighted fold-seed means and SDs, not confidence intervals or 15 independent datasets. Per-fold predictions and membership audits are exported.

| model | method | split | mae_mean | mae_sd | rmse_mean | rmse_sd | r2_mean | r2_sd | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph | fedavg | spatial | 68.1978 | 9.4667 | 96.1554 | 12.1115 | 0.4656 | 0.1948 | 15 |
| graph | reputation | spatial | 68.3153 | 9.3237 | 96.1869 | 11.7183 | 0.4656 | 0.1914 | 15 |
| graph | trust | spatial | 68.0117 | 8.8301 | 96.6926 | 11.3471 | 0.4651 | 0.1725 | 15 |
| graph | trimmed | spatial | 68.5532 | 9.3849 | 96.7137 | 11.7014 | 0.4593 | 0.1964 | 15 |
| graph | median | spatial | 68.6176 | 9.4978 | 96.7458 | 11.5549 | 0.4582 | 0.2001 | 15 |
| graph | krum | spatial | 68.6932 | 9.1172 | 97.2242 | 11.822 | 0.4527 | 0.2016 | 15 |
| graph | centralized | spatial | 71.1952 | 9.532 | 99.2511 | 11.3525 | 0.4341 | 0.1897 | 15 |
| spatial | kriging | spatial | 80.8413 | 19.1946 | 109.8027 | 25.108 | 0.2406 | 0.5004 | 15 |
| spatial | idw | spatial | 81.6914 | 19.2448 | 111.0735 | 24.9535 | 0.2244 | 0.5036 | 15 |

## Input calibration ablation

A separate experiment fits one linear least-squares correction per seen station using the first 10% of co-located noisy/reference pairs. The fitted coefficients correct later inputs and noisy training labels. Held-out locations do not fit calibration coefficients. All other settings match the original fixed-holdout experiment. This tests input correction rather than changing test targets or selecting hyperparameters on test results. It is not a full station-CV result.

| method | before | after | reduction_percent |
| --- | --- | --- | --- |
| fedavg | 98.107 | 78.491 | 19.995 |
| trust | 99.918 | 78.665 | 21.27 |
| centralized | 108.949 | 100.493 | 7.761 |

The corrected trust model reaches 78.67 ± 2.02 µg/m³ unseen-station RMSE, versus 99.92 ± 7.05 before correction. Corrected FedAvg reaches 78.49 ± 2.05. Thus input correction helps this simulation, while a benefit from trust weighting over FedAvg is still not established.

## Literature-anchored sensor sensitivity

Zheng et al. (2018), DOI 10.5194/amt-11-4823-2018, reports 35% and 46% one-hour ratio-error SD for calibrated Kanpur post-monsoon and monsoon measurements. We use these as relative-noise magnitudes in a separate mean-one lognormal multiplier: log SD = sqrt(log(1 + relative_SD²)), log mean = -0.5 log_SD². Choosing a lognormal distribution and assigning magnitudes to clients are our simulation assumptions. No hardware traces, temporal error correlation, or Delhi-specific coefficients are claimed. Drift, stuck blocks and dropout are zero in this scenario. The original additive/gain/annual-drift scenario remains a deliberately severe engineering stress test, not a Zheng-fitted model. The study's paired raw data are available on request, so field calibration remains unvalidated.

| scenario | method | split | rmse_mean | rmse_sd | mae_mean | mae_sd | r2_mean | r2_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kanpur_relative | centralized | spatial | 113.032 | 9.5406 | 76.9557 | 6.7764 | 0.3987 | 0.1016 |
| kanpur_relative | centralized | test | 95.6545 | 4.775 | 65.3713 | 2.7226 | 0.4783 | 0.0523 |
| kanpur_relative | fedavg | spatial | 100.2856 | 5.9691 | 68.8734 | 3.7272 | 0.5278 | 0.057 |
| kanpur_relative | fedavg | test | 99.2931 | 1.7712 | 67.4651 | 0.6294 | 0.4386 | 0.0201 |
| kanpur_relative | history_reputation | spatial | 100.3829 | 5.8692 | 68.9304 | 3.67 | 0.5269 | 0.0561 |
| kanpur_relative | history_reputation | test | 99.3273 | 1.7377 | 67.4812 | 0.6127 | 0.4383 | 0.0197 |
| kanpur_relative | median | spatial | 99.3164 | 7.6912 | 68.1724 | 4.8061 | 0.5361 | 0.0732 |
| kanpur_relative | median | test | 99.189 | 2.0734 | 67.3504 | 0.7607 | 0.4398 | 0.0236 |
| kanpur_relative | trust | spatial | 101.8592 | 5.3566 | 69.9571 | 3.5742 | 0.5131 | 0.0514 |
| kanpur_relative | trust | test | 99.7342 | 1.7389 | 67.7635 | 0.7655 | 0.4337 | 0.0198 |
| stress | history_reputation | spatial | 97.9293 | 4.2585 | 69.8834 | 5.0846 | 0.5499 | 0.0391 |
| stress | history_reputation | test | 93.7269 | 0.8231 | 65.8477 | 2.098 | 0.5002 | 0.009 |

## Reputation baseline

In addition to instantaneous inverse-distance update consistency, the history-reputation baseline maintains r_t = 0.8 r_(t-1) + 0.2 exp(-d_t / median(d_t)), initialized at 1. Here d is each client's update distance from the coordinate median; the denominator is floored at 1e-6. Aggregation uses sample_count × max(r, 0.05). This is a transparent project-defined temporal reputation comparator, not a reproduction of a named published algorithm. It is evaluated with three seeds in both sensor scenarios. Median, trimmed mean and Krum remain in the primary and station-CV suites. Krum assumes f=1; its guarantee does not apply when three or more clients are corrupted.

## Interactive features

Simulation Lab exposes bounded bias, gain, drift, Gaussian-noise and dropout controls. Its linear correction is fitted only to early co-located pairs, and its future-period RMSE is computed from valid paired hours. It does not retrain FL when sliders change. The lab also shows station coverage, missing hours, concentration summaries, actual PM10/NO2 monthly means and completed station CV. Existing mapping, client weights, convergence, robustness, privacy and exports remain available.

## Conclusion

In five station folds, uncorrected trust FL obtains 96.69 ± 11.35 µg/m³ RMSE and FedAvg 96.16 ± 12.11. Kriging obtains 109.80 ± 25.11; centralized graph training obtains 99.25 ± 11.35. The results support a working federated mapping simulation and the value of testing calibration, rather than an unconditional trust-weighting advantage. Prospective low-cost co-location, seasonal calibration transfer, secure aggregation and stronger privacy probes remain future work.
