# AirTrust demonstration guide

## Five minute walkthrough

1. Run `start.bat`. Open http://127.0.0.1:8000. Explain that this is a historical research simulation, using real Delhi reference observations and virtual low-cost sensor errors.
2. Overview: show the measured observation/station counts and the trust-versus-FedAvg result. Explain the actual outcome even if trust weighting is worse.
3. Pollution map: change the model, move the test-period hour slider and select Update map. The saved graph model computes a new surface. Select an open-circle held-out monitor to show its reference, simulated and test-prediction daily means.
4. Federation: show the eight virtual clients, calibration confidence and validation convergence. Explain that only parameter dictionaries and scalar weights enter the aggregation function. Show `models.py` if asked. Logical communication bytes are separate from actual network traffic.
5. Experiments: switch between known-station and unseen-station evaluation. Show mean ± SD over three seeds, the corruption sweep and exported per-run metrics. Explain why the temporal LSTM is absent from unseen-station results.
6. Privacy: show the update-based membership probe and explain AUC=0.5. State that FL alone does not guarantee privacy and that overlapping histories limit this test.
7. Datasets & exports: download the dataset ZIP and manifest. Show original Delhi/UCI source credit, CC BY licenses, reference-versus-simulated distinctions and exact processed samples. Open the implementation report for detailed evidence.

## Questions to be ready for

**What is actually federated?** Each geographic client's training examples stay in its local training routine. It receives a global parameter state, runs local optimization and returns a parameter state. The aggregator averages model parameters using selected weights. This is simulated on one machine, without real network or OS isolation.

**How is this different from FedAvg?** FedAvg weights by number of training examples. The proposed method multiplies that count by confidence from calibration residuals, then normalizes the weights. Equal averaging is a separate ablation and should not be called standard FedAvg.

**What is a trusted station?** Ground truth is a real reference monitor. A virtual noisy sensor is simulated at its exact location. Calibration pairs use only a prefix of seen training stations, preventing real spatial gradients from being mistaken for instrument error. Held-out stations never supply calibration scores.

**Can it map a place without a sensor?** The graph model uses other stations' recent histories, relative coordinates and time features. It does not require the target's reading or ID. We assess this at eight entire locations excluded from fitting. Dense-grid predictions have no direct labels and should not be presented as validated street-level measurements.

**What exactly is predicted?** PM2.5 concentration in µg/m³ one hour ahead, using six hours of history. The surface is PM2.5, not the official Indian AQI. The classical baselines use previous-hour donor readings to match this information timing.

**Why not use the LSTM for unseen stations?** Its input is the target's own history. At a location with no sensor that history is unavailable. The LSTM evaluates a separate known-station forecasting task.

**Why can trust weighting fail?** Downweighting a client's model update does not recalibrate that client's biased input readings at inference time. Client sample sizes, spatial heterogeneity, seasonal shift and stale calibration can also change utility. This is a measured hypothesis, not a guaranteed improvement.

**Why isn't centralized training always best?** Optimization and noisy-label bias affect it. Pooling data is a comparison condition, not a theorem that the chosen model and training budget reach the minimum possible error.

**What are the privacy guarantees?** None are formally certified. Individual model updates are exposed in the threat model and may leak. We measure one limited membership-ranking heuristic, not every possible attack. Secure aggregation and differential privacy remain future experiments.

**Did you use real low-cost sensor data?** No. Real reference observations form the ground truth; sensor errors are synthetic, with explicit numerical parameters. We cannot claim those settings reproduce a particular hardware sensor without paired field data.

**Why these datasets?** Downloadable historical station-level PM2.5 with coordinates supports time and spatial evaluation, and CC BY 4.0 permits attributed redistribution. Daily city-average AQI cannot validate a station-level pollution surface. Beijing supports a separate sanity benchmark and is not passed off as Delhi data.

**What should be submitted?** The two ZIPs, with the implementation report and measured figures as evidence. Existing course reports and slides remain separate originals. Publication evidence, an updated venue-format manuscript and any additional literature requirements must be handled honestly; this package does not claim acceptance or replace the team's academic review.

## Where to point in the code

- Sensor assumptions, held-out station exclusion and client ownership: `airtrust/data.py`.
- Confidence formula: the calibration calculation at the end of `prepare`.
- Spatial learner and aggregation equations: `airtrust/models.py`.
- Local training, checkpoint selection, attack and byte accounting: `airtrust/experiment.py`.
- True model-driven surface: `/api/map` in `airtrust/server.py`.
- Evidence against held-out leakage: `tests/test_core.py`.

## Operational notes

Do not stop the terminal serving the app during the demonstration. The packaged results can be used without retraining. Restart the app after regenerating experiments so cached seed data and results are consistent. The sources can be inspected from the dataset ZIP without external accounts. If the server reports a missing artifact, extract both ZIPs to the same parent folder and confirm `airtrust/datasets/raw` and `airtrust/results/models` both exist.
