# Team 10 requirements and evidence

Source: the supplied AY 2026–27 G2+TG2 BCSE316L Design of Smart Cities project workbook, all five sheets. This document maps evidence, not awarded marks. Teacher comments are treated as requested review corrections; historical documents are not evidence of experiments having run.

| Workbook sheet | Applicable requirement | Evidence / status |
|---|---|---|
| Students Option | Team 10 membership | README lists all four registrations and names |
| Titles Assigned | Team 10, Project 9; federated pollution mapping | Graph/LSTM learners, eight geographic clients, historical map and saved checkpoints |
| Titles Assigned | Heterogeneous quality and non-IID coverage | Geographic groups, explicit error profiles, trust and robust comparisons, corruption sweeps |
| Titles Assigned | Centralized and kriging alternatives | Primary metrics and five-fold spatial-CV exports |
| Titles Assigned | MAE/RMSE/R² and spatial error | Three-seed primary evaluation and five-fold station CV |
| Titles Assigned | Privacy/communication benefits | Measured probe and payload accounting. Benefits are not presumed; no formal privacy or universal byte saving is demonstrated |
| Titles Assigned | Calibration-aware aggregation contribution | Co-located virtual calibration, weighting comparison, input-correction ablation |
| Guidelines | At least 25 eligible papers; 70% recent; comparison table | LITERATURE_COMPARISON.csv and LITERATURE_REVIEW.md: 25 works, 20 from 2022 onward; foundational Zheng 2018 identified. Preprint-only and placeholder records excluded |
| Guidelines | Publication evidence | Not submitted. No acceptance, submission ID or venue-formatted submission is claimed |
| Review-1 Rubrics | Introduction, motivation and SDGs | README, implementation report and final manuscript |
| Review-1 Rubrics | Research gaps and measurable objectives | Literature synthesis; test calibration versus weighting, unseen locations, robustness and privacy/payload |
| Review-1 Rubrics | Baselines, architecture and feasible methods | README diagram, models.py, experiment.py and measured results |
| Review-1 Rubrics | Outcomes and exact metrics | Saved MAE/RMSE/R², runtime, parameters, payload and attack metrics |
| Review-1 Rubrics | Presentation and team Q&A | Demo guide prepared; actual participation and oral defense must be performed by the team |
| Review-2 Rubrics | Reproducible tools (8) | CPU PyTorch/FastAPI, dependency versions, offline data restoration, reproducibility scripts and tests |
| Review-2 Rubrics | Data description (6) | Sources, licenses, actual counts, cleaning, scaling, temporal/spatial splits and synthetic profiles |
| Review-2 Rubrics | Methodology (12) | Version-controlled modular code, baselines, architecture, hyperparameters and checkpoint selection |
| Review-2 Rubrics | Measured parameters (8) | Three seeds, mean/SD, explicit units, overhead and named comparators |
| Review-2 Rubrics | Analysis (12) | Data-drawn figures, CV, sensitivity, calibration ablation and explicit negative findings |
| Review-2 Rubrics | Conclusion/future work (6) | Quantified outcomes; prospective hardware calibration, stronger privacy and seasonal transfer remain limitations |
| Review-2 Rubrics | Publication (8) | External dissemination remains incomplete; this repository is not a publication acceptance |

## Teacher feedback corrections

| Issue | Evidence and qualification |
|---|---|
| No numerical results | Primary graph/LSTM, centralized, FedAvg, trust and spatial baselines over three seeds; metrics.csv and summary.csv |
| Need leave-stations-out CV | results/spatial_cv: five folds, every location held out once per seed, audits and predictions |
| Nearest-reference trust confounds gradients | Virtual sensors compared only with their own co-located reference during the initial calibration period |
| Robust/reputation baselines | Median, trimmed mean and Krum in primary/CV; instantaneous consistency plus temporal reputation in sensitivity. Temporal reputation is project-defined, not a named-paper replication |
| Unspecified sensor errors | Exact engineering stress parameters plus separate Kanpur relative-error sensitivity. Lognormal generation is an explicit assumption. No real Delhi low-cost calibration fit is claimed |
| Illustrative station map and volume figure | Actual 36 retained coordinates; source and retained counts generated from files |
| Dataset roles | Reference-grade Delhi labels, simulated low-cost clients, contextual PM10/NO₂ and separate Beijing persistence benchmark |
| Broken privacy reference and threat model | Named individual-update loss-improvement attack, server threat, AUC and advantage; no formal guarantee |
| Unsupported figures/disclosure | Figures generated from data; AI assistance disclosed |
| Keyword, email, abbreviations and grammar | Updated manuscript includes federated learning, supplied student email and expanded LSTM/GNN; prose revised |

## Boundaries relevant to grading

This is a simulation using public data, not a multi-device field deployment. Sensor coefficients require prospective field validation. The mapped rectangle has no grid-cell ground truth or administrative-boundary accuracy claim. Neither the basic trust model nor the extended CV establishes trust superiority over FedAvg. A venue submission, acceptance and live team demonstration cannot be generated as experimental evidence by software.
