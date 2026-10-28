# AirTrust Delhi

Working implementation of Team 10's BCSE316L project: **Trust-Weighted Federated Learning for City-Wide Air Pollution Mapping in Delhi**.

The project includes real Delhi station observations, explicitly simulated low-cost sensor errors, temporal and graph learners, federated and centralized comparisons, geostatistical baselines, an empirical update-based membership probe, a local dashboard, trained models and reproducible results. Original course submissions in the parent folder are preserved.

## Start the demonstration

On this computer, double-click **start.bat** or run:

```powershell
cd "C:\Users\Rishit\Desktop\7th SEM\smart city project\airtrust"
python -m uvicorn airtrust.server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Keep the terminal running; Ctrl+C stops the server. If port 8000 is already used, select another port in the command and open that port. The dashboard uses local assets and works offline after dependencies and supplied datasets are present.

## Install on another computer

Extract **both** submission ZIPs into the same parent directory. Each ZIP contains an `airtrust/` directory, so the code and datasets merge into one project directory. Python 3.13 was used for the measured run. Python 3.11 or later should work with compatible dependencies; exact numerical reproduction is best attempted with the recorded versions.

```powershell
cd airtrust
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn airtrust.server:app --host 127.0.0.1 --port 8000
```

`requirements-lock.txt` records the actual tested versions. The measured run used CPU PyTorch. If the default PyTorch install is unsuitable for your platform, use the official CPU wheel instructions at https://pytorch.org/get-started/locally/ . No API key, paid service, GPU, database, or cloud account is needed for the supplied demonstration.

## Reproduce everything

```powershell
# Only needed if source files are absent; requires internet.
python scripts/download_data.py

# Measured settings: three seeds, 20 rounds, 12 local steps; includes corruption sweep.
python scripts/run_experiments.py

# Meaningful algorithm, data-isolation and API tests.
python -m pytest

# Export processed datasets, figures, report and both submission ZIPs.
python scripts/build_submission.py

# Check all source and derived dataset SHA-256 checksums.
python scripts/verify_submission.py
```

On Windows, `reproduce.bat` runs training, tests and packaging using the local virtual environment if present. An initial smoke run is available through `python scripts/run_experiments.py --seeds 17 --rounds 2 --steps 2 --no-robustness --output results_smoke`; smoke numbers are not the reported results. Runtime depends on hardware. Running results generation while the dashboard serves old results may temporarily mix artifact generations; restart or reload after completion.

## Project layout

| Path | Purpose |
| --- | --- |
| `airtrust/data.py` | Cleaning, distinct station locations, synthetic errors, chronological/spatial splits, causal histories |
| `airtrust/models.py` | Temporal LSTM, inductive spatial graph model and aggregation algorithms |
| `airtrust/experiment.py` | Repeated-run training, comparisons, privacy probe, corruption sweep and metrics |
| `airtrust/server.py` | FastAPI endpoints and saved-model spatial inference |
| `web/` | Local dashboard with model/time/station controls and exports |
| `datasets/raw/` | Original licensed data, metadata and API response chunks |
| `datasets/processed/` | Cleaned reference data, seeded virtual clients, exact samples and splits |
| `datasets/manifest.json` | Original/derived file sizes, provenance and SHA-256 hashes |
| `results/models/` | Validation-selected graph and LSTM checkpoints |
| `results/metrics.csv` | Per-seed MAE, RMSE, R² and sample counts |
| `results/summary.csv` | Means and sample standard deviations |
| `results/predictions.csv.gz` | Actual held-out predictions, station IDs, timestamps and reference labels |
| `results/robustness.csv` | Corruption-share sweep |
| `results/privacy.csv` | Individual-update membership probe |
| `results/communication.csv` | Parameter payload counts and simulation runtime |
| `results/figures/` | Programmatically generated experiment figures |
| `docs/IMPLEMENTATION_REPORT.md` | Actual methods, measured results, limitations and attribution |
| `docs/DEMO_GUIDE.md` | Demonstration sequence and viva explanations |
| `submission/datasets_used.zip` | Separate dataset submission |
| `submission/airtrust_project.zip` | Code, models, results and documentation; pair with dataset ZIP |

## Interpret the results correctly

The task is **one-hour-ahead spatial prediction** using other sensors' six-hour histories, plus a separate target-history temporal forecasting task. The graph model never reads its target station's history. Eight complete locations are excluded from fitting, calibration and donor inputs. The temporal LSTM is not evaluated at unseen stations. Compare graph methods against each other; LSTM scores have a richer information set.

Eight ownership/geographic clients run in a single Python process. Data remain local to client training routines by algorithm design, but process isolation and real distributed networking are not implemented. The simulator/evaluator loads public ground truth. Trust scores compare co-located virtual/reference pairs during a calibration-only prefix. The checkpoint selector also has clean validation references; this is an explicitly disclosed source of trusted feedback.

Bias, gain, drift, noise, dropouts and stuck readings are **synthetic experimental assumptions**, not measurements from real deployed low-cost sensors. Weighting may lose to FedAvg or classical baselines; the report preserves those findings. There is no formal privacy guarantee, validated street-level accuracy, or blanket communication-saving claim. Krum is an empirical comparator outside its theoretical corruption bound in the main scenario. The Beijing data support a separate persistence benchmark only.

## Dataset credit

Delhi: **India Air Quality Database, XKDR Forum**, based on **Central Pollution Control Board (CAAQM)** observations. https://github.com/xKDR/Air-Quality-Database . Compilation license: **CC BY 4.0**.

Beijing: **Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository.** https://doi.org/10.24432/C5RK5G . License: **CC BY 4.0**.

See `datasets/README.md` for roles, fields, modifications, licensing, source coverage and redistribution attribution. Dataset terms are separate from the source-code license. `docs/IMPLEMENTATION_REPORT.md` provides method references and an accurate generative-AI disclosure. No publication or acceptance is claimed.
