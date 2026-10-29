"""Create traceable figures, evidence notes and offline public-data bundle."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
from build_submission import md_table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    datasets=ROOT/'datasets';results=ROOT/'results';docs=ROOT/'docs';figs=results/'figures'
    expansion=json.loads((datasets/'expansion.json').read_text())
    manifest=json.loads((datasets/'manifest.json').read_text());manifest['expansion']=expansion;manifest['files']=[]
    for p in sorted((datasets/'raw').rglob('*')):
        if p.is_file():manifest['files'].append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (datasets/'manifest.json').write_text(json.dumps(manifest,indent=2))
    bundle=datasets/'bundled';bundle.mkdir(exist_ok=True);target=bundle/'source_data.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted((datasets/'raw').rglob('*')):
            if p.is_file():z.write(p,p.relative_to(datasets))
    (bundle/'manifest.json').write_text(json.dumps(dict(file=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),license='CC BY 4.0',source_manifest='../manifest.json'),indent=2))
    cv=pd.read_csv(results/'spatial_cv'/'summary.csv');cal=pd.read_csv(results/'calibration'/'summary.csv');sense=pd.read_csv(results/'sensitivity'/'summary.csv')
    spatial=cv[cv.split=='spatial'].sort_values('rmse_mean');plt.rcParams.update({'font.size':10,'figure.dpi':150})
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(spatial.method,spatial.rmse_mean,xerr=spatial.rmse_sd,color='#357b64');ax.set_xlabel('Unseen-station RMSE (µg/m³), mean ± SD over 5 folds × 3 seeds');ax.invert_yaxis();fig.tight_layout();fig.savefig(figs/'spatial_cv.png');plt.close(fig)
    primary=pd.read_csv(results/'summary.csv');rows=[]
    for method in ['fedavg','trust','centralized']:
        before=primary.query("model=='graph' and split=='spatial' and method==@method").iloc[0]
        after=cal.query("model=='graph' and split=='spatial' and method==@method").iloc[0]
        rows.append(dict(method=method,before=before.rmse_mean,after=after.rmse_mean,reduction_percent=100*(before.rmse_mean-after.rmse_mean)/before.rmse_mean))
    comparison=pd.DataFrame(rows);comparison.to_csv(results/'calibration'/'comparison.csv',index=False)
    ax=comparison.set_index('method')[['before','after']].plot.bar(color=['#bd895d','#357b64'],rot=0,figsize=(7,4));ax.set_ylabel('Unseen-station RMSE (µg/m³)');ax.set_xlabel('Same fixed station holdout, three seeds');ax.figure.tight_layout();ax.figure.savefig(figs/'calibration_ablation.png');plt.close(ax.figure)
    report='''# Extended evaluation and dataset expansion

## Data and scope

The Delhi source now contains 966,067 hourly pollutant observations: 317,627 PM2.5, 323,061 PM10 and 325,379 NO2 rows from 39 catalog entries in 2024. This adds 648,440 observations, a 204.2% increase over the original Delhi source. The PM2.5 experiment retains the original 36 distinct stations and 300,355 valid labels. Additional pollutant rows support context exploration, not extra PM2.5 training labels. Beijing contributes 420,768 station-hour source rows across 12 stations. These counts exclude duplicate ZIP representations, weekly provenance copies and synthetic derivatives.

## Spatial cross-validation

Five deterministic station folds rotate through each geographic client. Every retained location is unseen exactly once per seed. Four folds hold out eight locations and the fifth holds out four. Donors, training labels, validation labels and calibration exclude all stations in the current holdout. Future-time test boundaries are unchanged. Three seeds (17, 29, 43), 20 rounds, 12 local steps and Adam 0.002 are used throughout. Summary values below are unweighted fold-seed means and SDs, not confidence intervals or 15 independent datasets. Per-fold predictions and membership audits are exported.

'''
    report+=md_table(spatial.round(4))+'\n\n'
    report+='''## Input calibration ablation

A separate experiment fits one linear least-squares correction per seen station using the first 10% of co-located noisy/reference pairs. The fitted coefficients correct later inputs and noisy training labels. Held-out locations do not fit calibration coefficients. All other settings match the original fixed-holdout experiment. This tests input correction rather than changing test targets or selecting hyperparameters on test results. It is not a full station-CV result.

'''+md_table(comparison.round(3))+'\n\n'
    report+='''The corrected trust model reaches 78.67 ± 2.02 µg/m³ unseen-station RMSE, versus 99.92 ± 7.05 before correction. Corrected FedAvg reaches 78.49 ± 2.05. Thus input correction helps this simulation, while a benefit from trust weighting over FedAvg is still not established.

## Literature-anchored sensor sensitivity

Zheng et al. (2018), DOI 10.5194/amt-11-4823-2018, reports 35% and 46% one-hour ratio-error SD for calibrated Kanpur post-monsoon and monsoon measurements. We use these as relative-noise magnitudes in a separate mean-one lognormal multiplier: log SD = sqrt(log(1 + relative_SD²)), log mean = -0.5 log_SD². Choosing a lognormal distribution and assigning magnitudes to clients are our simulation assumptions. No hardware traces, temporal error correlation, or Delhi-specific coefficients are claimed. Drift, stuck blocks and dropout are zero in this scenario. The original additive/gain/annual-drift scenario remains a deliberately severe engineering stress test, not a Zheng-fitted model. The study's paired raw data are available on request, so field calibration remains unvalidated.

'''+md_table(sense.round(4))+'\n\n'
    report+='''## Reputation baseline

In addition to instantaneous inverse-distance update consistency, the history-reputation baseline maintains r_t = 0.8 r_(t-1) + 0.2 exp(-d_t / median(d_t)), initialized at 1. Here d is each client's update distance from the coordinate median; the denominator is floored at 1e-6. Aggregation uses sample_count × max(r, 0.05). This is a transparent project-defined temporal reputation comparator, not a reproduction of a named published algorithm. It is evaluated with three seeds in both sensor scenarios. Median, trimmed mean and Krum remain in the primary and station-CV suites. Krum assumes f=1; its guarantee does not apply when three or more clients are corrupted.

## Interactive features

Simulation Lab exposes bounded bias, gain, drift, Gaussian-noise and dropout controls. Its linear correction is fitted only to early co-located pairs, and its future-period RMSE is computed from valid paired hours. It does not retrain FL when sliders change. The lab also shows station coverage, missing hours, concentration summaries, actual PM10/NO2 monthly means and completed station CV. Existing mapping, client weights, convergence, robustness, privacy and exports remain available.

## Conclusion

In five station folds, uncorrected trust FL obtains 96.69 ± 11.35 µg/m³ RMSE and FedAvg 96.16 ± 12.11. Kriging obtains 109.80 ± 25.11; centralized graph training obtains 99.25 ± 11.35. The results support a working federated mapping simulation and the value of testing calibration, rather than an unconditional trust-weighting advantage. Prospective low-cost co-location, seasonal calibration transfer, secure aggregation and stronger privacy probes remain future work.
'''
    (docs/'EXTENDED_EVALUATION.md').write_text(report,encoding='utf-8')
    print('Built extensions and source bundle',target.stat().st_size,flush=True)
if __name__=='__main__':main()

