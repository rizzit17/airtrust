# Evaluation demonstration

Allow about 10 minutes. Start with the saved results; full retraining can be demonstrated separately.

## Plain-language teaching mode

Open the public site and press **Explain simply** in the top bar. Every screen
gets a short analogy and a “During the demo” speaking cue. Underlined technical
terms open a definition, analogy, suggested explanation and an important
distinction. **Term guide** searches 64 explanations, including database/file
formats, FL, GNN, LSTM, RMSE, MAE, R², folds, trust and privacy. The toggle is
remembered in this browser; turning it off restores the regular view.

For a short demonstration:

1. Start at Overview: explain the eight study groups keeping their notebooks and
   sharing learned rules. State that these are simulated groups on one computer.
2. Open Simulation lab and press **Try the ruler story ↓**.
3. Keep **CRRI Mathura Road** selected. Press **1 · A perfect ruler**: RMSE is
   0.0 → 0.0. Say, “With no added sensor error, there is nothing to correct.”
4. Press **2 · A faulty ruler**: 59.4 → 18.1. Say, “The early ruler check reduces
   later error, although it does not remove every error.”
5. Press **3 · It drifts further**: 114.0 → 64.1. Say, “A correction learned early
   can become stale when the sensor changes more later.”
6. Click an underlined **RMSE**, **calibration**, or **drift** for a short
   explanation. Results use actual API calculations, not hard-coded displays.
7. Explain that these are sensor-reading diagnostics. The saved FL models are
   not retrained by these buttons. Experiment tables remain the model evidence.

The presets populate the existing controls and run the same scenario endpoint.
They do not change the experimental splits, saved models or evaluation results.
At another station the scores can differ. All analogies simplify the explanation;
they do not establish additional accuracy, privacy or deployment guarantees.

## Full review sequence

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
