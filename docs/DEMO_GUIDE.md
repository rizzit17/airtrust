# Evaluation demonstration

Allow about 10 minutes. Start with the saved results; full retraining can be demonstrated separately.

1. Restore data and start the app using the README. Show Overview and explain real reference data versus virtual low-cost measurements.
2. Open Pollution map. Change the model and test hour, then inspect a held-out station. Explain why interpolated grid cells have no directly observed truth.
3. Open Federation. Explain early co-location RMSE, geographic client groups, confidence weights and checkpoint selection.
4. Open Experiments. Contrast LSTM known-station forecasting with graph unseen-location prediction; show mean and SD and the corruption sweep.
5. Open Simulation lab. Inspect completed station CV and station coverage. Run a zero-bias, gain-one, zero-drift/noise/dropout scenario: raw RMSE should be zero. Then introduce drift and explain why an early correction can become stale.
6. Show PM10/NO₂ monthly observations. They expand contextual data, not PM2.5 training labels.
7. Show Privacy probe and communication accounting. Explain the specified attack, correlation limitation and absence of a formal guarantee.
8. Open EXTENDED_EVALUATION.md, the literature comparison and REQUIREMENTS.md. Discuss the actual negative trust result and improved input-calibration result.
9. Show `python -m pytest -q`, source manifest and reproducibility commands. Do not claim the team has already delivered the oral presentation or obtained publication acceptance.

Suggested speaking coverage: Rishit—problem and application; Siddharth—data and literature; Jashan—models and evaluation; Adarsh—privacy, limitations and demonstration. This is a proposed allocation, not a record of individual contributions.

## Questions to prepare

- Why can local training still leak information? Model updates can encode training information; raw-data locality is not differential privacy.
- Why use virtual sensors? Public reference data provide controlled labels; real co-located low-cost traces are not available in this project.
- Why not report the LSTM as the best mapper? It consumes a target station's own history, so it does not solve completely unseen-location mapping.
- Why does trust not always win? Weighting reduces influence of some clients but does not correct biased donor features; geographic diversity may also be lost.
- What does input calibration change? Only noisy seen-station inputs/labels using early paired observations; test reference truth is unchanged.
- What is still required for deployment? Field co-location, geographic/seasonal transfer checks, secure protocols, real network measurements and prospective evaluation.
