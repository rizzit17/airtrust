# Hosting AirTrust on Vercel

**Production:** https://airtrust.vercel.app — deployed and checked on 29 October 2026.
Open https://airtrust.vercel.app/#lab for the Simulation Lab. No Vercel sign-in is
required at the production URL. The unique deployment preview URL can require
authentication; share the production URL above with reviewers.

The hosted dashboard runs the same Python scenario calculations and saved spatial
graph models as the local application. Training remains an offline research task.
No models are retrained when a visitor changes a Simulation Lab control.

## Build and verify

From the repository root with the normal project dependencies installed:

```bash
python scripts/prepare_local_data.py
python scripts/package_submission.py
python scripts/build_hosting.py
python scripts/verify_hosting.py
python -m pytest -q
```

The generated `deployment/vercel/` directory is the deployment root. It includes
the frontend, FastAPI application, measured result tables, compact runtime data,
exported model weights, and downloadable submission packages. Generated copies
are ignored by Git; the builder and inference implementation are versioned.

The source dependency list is intentionally different from the training list:
NumPy, pandas, FastAPI and Uvicorn are sufficient for serving this application.
`NumpySpatialGraph` evaluates the trained message layers, attention, residual and
head using the exported checkpoint weights. Twelve equivalence tests compare it
with the original PyTorch models; no architecture or parameter values are changed.

Runtime snapshots preserve the reference, simulated sensor and causally filled
input arrays for seeds 17, 29 and 43. They omit the unused training-sample tensors,
so a web request does not construct the full research training dataset.
Snapshots retain the same station split and calibration boundaries.

## Deploy

Sign in to your own Vercel account, then deploy the generated directory:

```bash
vercel login
cd deployment/vercel
vercel deploy --prod --yes --name airtrust
```

The generated configuration selects the FastAPI framework and a 60-second
function duration. A Python 3.13 environment is requested. Keep the deployment
directory when publishing an update so its `.vercel/` project linkage is reused.
The CLI prints the actual production URL after a successful deployment; do not
assume that a particular `airtrust.vercel.app` name is available.

Use a free personal project where eligible. No paid plan, database, Blob storage,
or external API subscription is required by this implementation.

## Downloads and limits

Large exports are served as static CDN files under `public/downloads/`, with the
API redirecting to them. This avoids routing ZIPs through Vercel's function
response-size limit. The individual PM2.5 CSV is gzip-compressed for the hosted
download; its decompressed contents are unchanged. Full datasets and project
submission ZIPs remain available.

The package is about 93 MB and is checked against the 100 MB Hobby CLI source
upload limit. Refreshing submission archives can change its size; rerun
`verify_hosting.py` before an update. The first request after a cold start can be
slower. This is a historical research simulation with live model inference, not
a feed of live sensors or a multi-device federation deployment.

## Post-deployment checks

Open the production URL without a Vercel login. Verify the Overview, map update,
station chart, Experiments, Simulation Lab scenario controls and dataset exports.
Verify `/api/map?hour=8000&method=trust`, `/api/context/pm10`, and
`/api/scenario?station_id=site_103` return successful JSON responses.

Production verification returned HTTP 200 for the dashboard, frontend script,
overview, stations, graph map, scenario, contextual pollutant, extended results,
and quality endpoints. The graph map returned 784 grid points. The default CRRI
scenario returned RMSE 59.3733 → 18.0898 on 1,213 test pairs, matching the local
calculation. The hosted browser scenario was also run successfully. All eleven
download exports were checked for public accessibility and matching content.

Platform references: [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi),
[function limits](https://vercel.com/docs/functions/limitations), and
[CLI source upload limits](https://vercel.com/docs/limits).
