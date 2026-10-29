# Datasets used by AirTrust

## Attribution and license

Delhi: India Air Quality Database, XKDR Forum, and Central Pollution Control Board (CAAQM network). Compilation license: Creative Commons Attribution 4.0 International. https://github.com/xKDR/Air-Quality-Database and https://airquality.xkdr.org . Only CPCB observations are requested; no AirNow source rows are used.

Beijing: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. DOI https://doi.org/10.24432/C5RK5G . License: Creative Commons Attribution 4.0 International.

License terms: https://creativecommons.org/licenses/by/4.0/ . Provide credit, a link to the license and indicate modifications when redistributing. No endorsement is implied. The compilation and all derived data are provided without warranty. Source data are not a certified health advisory.

## Exact data used

raw/delhi_pm25_hourly_2024.csv: 317,627 source rows; 39 source station entries; 2024 calendar year, hourly PM2.5, local IST. Columns: station_id, parameter_name, unit, period_start, mean, min, max, n. Original API responses are also retained in raw/delhi_hourly; these are provenance copies, not extra independent observations. Source metadata and coordinates are in raw/delhi_stations.json. Download route: /v1/measurements with city=Delhi, source=cpcb_caaqm, parameter=PM2.5, agg=hourly and seven-day windows to stay below the public demo row cap.

raw/beijing_multi_site_uci.zip: unmodified UCI source bundle. raw/beijing_station_data.zip: nested station CSV archive extracted from that bundle. These are duplicate representations of the same dataset. It has 420,768 reference station-hour rows across 12 stations and 18 source columns. The secondary benchmark uses PM2.5 only. Nested CSVs can be extracted with any ZIP utility.

processed/delhi_reference_cleaned_2024.csv.gz: 8,784 hours × 36 retained station columns, with missing values preserved. Valid values: 300,355. Coverage cutoff 70%; values outside 0–2,000 µg/m³ excluded; entries within 50 m consolidated by keeping the most complete station. Original values otherwise unchanged.

processed/seed_*/client_*_simulated_pm25.csv.gz: virtual client readings (µg/m³), including held-out stations as separately identifiable evaluation-only sensors. Ownership does not imply fitting membership. station_assignments.csv identifies client and held-out status. Those held-out records are excluded by all model training, calibration and donor builders. These files are **synthetic derivatives of real reference observations**, not independent real low-cost sensor measurements. All simulation parameters and exact splits are in simulation_and_split.json.

processed/seed_*/*_samples.npz: exact prepared samples. Fields: x [sample,3 donors,8 features] (six PM2.5 lags divided by 100, relative x and y kilometers divided by 50); own [sample,6] target histories divided by 100 for the temporal task; context [sample,4] hourly/yearly sine/cosine; y noisy target divided by 100; truth unimputed reference divided by 100; t integer hour index; s station index in station_assignments.csv; client zero-based client. train/nonmember sets are disjoint target-window pairs. Models consume x or own, context and noisy y only. Evaluator consumes truth. NPZ arrays intentionally include evaluator truth and target histories even in the spatial split; the graph model never consumes own or truth during fitting/inference.

## Source roles and exclusions

Delhi is the primary federated experiment. Beijing is a separate chronological temporal persistence benchmark; it is not a second federated evaluation. OpenAQ, sensor.community, meteorology services and Kaggle are not used in these results. There are no private individual records.

## Reproduce and verify

Run scripts/download_data.py to retrieve sources, scripts/run_experiments.py to generate results and scripts/build_submission.py to export derived files. Verify every entry in manifest.json with SHA-256. Manifest timestamps record actual download time in UTC, while observation times remain in source local time. The application needs no source API access once the files are present. CC BY attribution applies to source and modified derived data; this README identifies the modifications.

## Expanded contextual observations

datasets/raw/delhi_pm10_hourly_2024.csv: 323,061 observed PM10 rows at 39 Delhi catalog entries, hourly IST in 2024. CC BY 4.0, India Air Quality Database / XKDR Forum / CPCB CAAQM. Used for contextual monthly trends, not PM2.5 training labels.

datasets/raw/delhi_no2_hourly_2024.csv: 325,379 observed NO2 rows at 39 Delhi catalog entries, hourly IST in 2024. CC BY 4.0, India Air Quality Database / XKDR Forum / CPCB CAAQM. Used for contextual monthly trends, not PM2.5 training labels.

Delhi totals 966,067 pollutant rows across three parameters; added contextual rows total 648,440. Including the 420,768 Beijing station-hour rows gives 1,386,835 source rows with differing schemas. Duplicated provenance copies and virtual sensor derivatives are excluded from these counts.

The GitHub source snapshot is bundled/source_data.zip with its SHA-256 in bundled/manifest.json. Run scripts/prepare_local_data.py to restore the exact source tree offline, then scripts/build_submission.py to generate processed samples and submission ZIPs. Extended experiment audits and per-fold metrics are included under results/.
