"""Export datasets, data-derived figures, a measured report, and submission ZIPs."""
import sys, json, hashlib, zipfile, platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from airtrust.data import ROOT,prepare

LABELS={'trust':'Trust-weighted FL','fedavg':'FedAvg','centralized':'Centralized','equal':'Equal averaging','median':'Coordinate median','trimmed':'Trimmed mean','krum':'Krum (f=1)','reputation':'Update consistency','kriging':'Ordinary kriging','idw':'IDW','persistence':'Persistence'}

def md_table(df):
    vals=df.astype(str).values.tolist()
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in vals)

def main():
    results=ROOT/'results'; docs=ROOT/'docs'; figs=results/'figures'; processed=ROOT/'datasets'/'processed'
    for p in [docs,figs,processed,ROOT/'submission']:p.mkdir(parents=True,exist_ok=True)
    run=json.loads((results/'run.json').read_text()); summary=pd.read_csv(results/'summary.csv'); allmetrics=pd.read_csv(results/'metrics.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    data=prepare(17); a=data['audit']; stations=data['stations']; ref=data['reference']; noisy=data['noisy']
    for seed in run['seeds']:
        d=data if seed==17 else prepare(seed)
        folder=processed/f'seed_{seed}';folder.mkdir(exist_ok=True)
        assignments=d['stations'].copy(); assignments['client']=d['groups']; assignments['held_out']=assignments.index.isin(d['audit']['heldout_stations']);assignments.to_csv(folder/'station_assignments.csv')
        for g in range(8):
            ids=np.where(d['groups']==g)[0]
            df=pd.DataFrame(d['noisy'][:,ids],columns=d['stations'].index[ids],index=d['times'])
            df.index.name='timestamp_IST'
            df.to_csv(folder/f'client_{g+1:02d}_simulated_pm25.csv.gz',compression='gzip')
        # Exact tensors and membership assignments used by the learner/evaluator.
        for name,split in d['splits'].items(): np.savez_compressed(folder/f'{name}_samples.npz',**split)
        (folder/'simulation_and_split.json').write_text(json.dumps(d['audit'],indent=2),encoding='utf-8')
    clean=pd.DataFrame(ref,columns=stations.index,index=data['times']);clean.index.name='timestamp_IST';clean.to_csv(processed/'delhi_reference_cleaned_2024.csv.gz',compression='gzip')
    fig,ax=plt.subplots(figsize=(8,6))
    seen=~stations.index.isin(a['heldout_stations'])
    for g in range(8):
        mask=(data['groups']==g)&seen;ax.scatter(stations.longitude[mask],stations.latitude[mask],s=40,label=f'Client {g+1}')
    ax.scatter(stations.longitude[~seen],stations.latitude[~seen],s=90,facecolors='none',edgecolors='black',linewidths=1.5,label='Unseen test stations')
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Delhi monitoring coordinates and simulated geographic clients')
    ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2);fig.savefig(figs/'station_map.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));counts=pd.Series(np.isfinite(ref).sum(0),index=stations.station_name.str.split(',').str[0]).sort_values();counts.plot.bar(ax=ax,color='#4b8260');ax.set(ylabel='Valid hourly PM2.5 observations',title='Actual retained observations per station, calendar year 2024');ax.tick_params(axis='x',labelsize=7);fig.savefig(figs/'record_counts.png');plt.close(fig)
    for split,title in [('spatial','Unseen stations in the future test period'),('test','Known stations in the future test period')]:
        s=summary[(summary.split==split)&(summary.model.isin(['graph','spatial']))].sort_values('rmse_mean')
        fig,ax=plt.subplots(figsize=(9,5));ax.barh([LABELS[m] for m in s.method],s.rmse_mean,xerr=s.rmse_sd, color=['#2e7854' if m=='trust' else '#b9cbb0' for m in s.method],capsize=3);ax.invert_yaxis();ax.set(xlabel='RMSE (µg/m³), mean ± sample SD across three seeds',title=title);ax.grid(axis='x',alpha=.2);fig.savefig(figs/f'comparison_{split}.png');plt.close(fig)
    hist=json.loads((results/'history.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4))
    for m in ['trust','fedavg','centralized','median']:
        h=hist[f'17_graph_{m}'];ax.plot([r['round'] for r in h],[r['validation_rmse'] for r in h],label=LABELS[m])
    ax.set(xlabel='Round',ylabel='Validation RMSE (µg/m³)',title='Validation convergence, seed 17');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(figs/'convergence.png');plt.close(fig)
    sweep=pd.read_csv(results/'robustness.csv');fig,ax=plt.subplots(figsize=(8,4))
    for m in ['trust','fedavg','median','trimmed']:
        ss=sweep[(sweep.method==m)&(sweep.split=='spatial')].groupby('actual_corruption').rmse.agg(['mean','std'])
        ax.errorbar(ss.index*100,ss['mean'],yerr=ss['std'],marker='o',capsize=3,label=LABELS[m])
    ax.set(xlabel='Corrupted clients (%)',ylabel='Unseen-station RMSE (µg/m³)',title='Robustness to specified synthetic sensor errors');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(figs/'robustness.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));profiles=a['profiles'];weights=np.array(hist['17_graph_trust'][0]['weights']);ax.bar(np.arange(8)-.18,[p['confidence'] for p in profiles],.36,label='Calibration confidence');ax.bar(np.arange(8)+.18,weights,.36,label='Normalized aggregation weight');ax.set(xticks=np.arange(8),xticklabels=[f'C{i+1}' for i in range(8)],ylabel='Score / weight',title='Trust and sample-adjusted aggregation weights, seed 17');ax.legend();fig.savefig(figs/'trust_weights.png');plt.close(fig)
    privacy=pd.read_csv(results/'privacy.csv');pr=privacy[privacy.model=='graph'].groupby('method').auc.agg(['mean','std'])
    fig,ax=plt.subplots(figsize=(8,4));ax.bar([LABELS[m] for m in pr.index],pr['mean'],yerr=pr['std'],capsize=3,color='#9eb892');ax.axhline(.5,color='#a75c4d',linestyle='--',label='Chance ranking');ax.set(ylabel='Membership-inference AUC',ylim=(0,1),title='Individual-update loss-improvement probe');ax.tick_params(axis='x',labelrotation=30);ax.legend();fig.savefig(figs/'privacy_probe.png');plt.close(fig)
    # Actual prediction errors at unseen station coordinates, not invented area truth.
    pred=pd.read_csv(results/'predictions.csv.gz');errors=pred[(pred.model=='graph')&(pred.method=='trust')&(pred.split=='spatial')&(pred.seed==17)].assign(squared=lambda q:(q.prediction-q.truth)**2).groupby('station_id').squared.mean().pow(.5)
    es=stations.loc[errors.index];fig,ax=plt.subplots(figsize=(7,5));sc=ax.scatter(es.longitude,es.latitude,c=errors,s=100,cmap='YlOrRd');fig.colorbar(sc,ax=ax,label='RMSE (µg/m³)');ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Measured error at unseen Delhi stations, seed 17');fig.savefig(figs/'spatial_errors.png');plt.close(fig)
    # A map is an inference product; no dense-grid ground truth is available.
    from airtrust.server import pollution_map
    mapped=pollution_map(8000,'trust',17)
    grid=pd.DataFrame(mapped['points']);grid.to_csv(results/'map_snapshot.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,6));sc=ax.scatter(grid.longitude,grid.latitude,c=grid.pm25,s=45,marker='s',cmap='YlOrRd');fig.colorbar(sc,ax=ax,label='Predicted PM2.5 (µg/m³)');ax.scatter(stations.longitude,stations.latitude,s=12,color='black',label='Monitoring location');ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title=f'Graph FL prediction surface\n{mapped["timestamp"]} IST');ax.legend();fig.savefig(figs/'pollution_surface.png');plt.close(fig)
    chosen=summary[(summary.model=='graph')&(summary.split=='spatial')].set_index('method');tw=chosen.loc['trust'];fa=chosen.loc['fedavg'];best=summary[(summary.split=='spatial')].sort_values('rmse_mean').iloc[0];gain=100*(fa.rmse_mean-tw.rmse_mean)/fa.rmse_mean
    tables={}
    for split in ['spatial','test']:
        s=summary[summary.split==split].copy();t=pd.DataFrame({'Model':s.model,'Method':s.method.map(LABELS),'RMSE mean ± SD':s.apply(lambda r:f'{r.rmse_mean:.2f} ± {r.rmse_sd:.2f}',axis=1),'MAE mean ± SD':s.apply(lambda r:f'{r.mae_mean:.2f} ± {r.mae_sd:.2f}',axis=1),'R² mean ± SD':s.apply(lambda r:f'{r.r2_mean:.3f} ± {r.r2_sd:.3f}',axis=1)})
        tables[split]=md_table(t)
    cost=pd.read_csv(results/'communication.csv');graphcost=cost[(cost.model=='graph')&(cost.method=='trust')];upload=float(graphcost.upload_bytes.mean());rawbytes=float(graphcost.raw_training_bytes.mean())
    movement=(upload*2/rawbytes)
    profiletable=pd.DataFrame({'Scenario':['Mild','Severe'],'Bias µg/m³':[1.5,35],'Gain':[1.02,1.25],'Drift over year µg/m³':[2,30],'Noise SD µg/m³':[3,12],'Dropout probability':[.01,.04],'Six-hour stuck blocks':[0,12]})
    ptable=privacy.groupby(['model','method']).agg(auc_mean=('auc','mean'),auc_sd=('auc','std'),advantage_mean=('advantage','mean')).reset_index();ptable['method']=ptable.method.map(LABELS);ptable=ptable.round(4)
    report=f'''# Trust Weighted Federated Learning for City Wide Air Pollution Mapping in Delhi

BCSE316L Design of Smart Cities · Team 10 · Project 9 · AY 2026–27

Rishit Chaudhary 23BCT0157; Siddharth Verma 23BCT0159; Jashan Sahasrabudhe 23BCT0022; Adarsh Narayan 23BCT0155.

## Abstract

We implemented and evaluated a single-machine federated air-quality simulation using real hourly Delhi reference observations and explicitly simulated low-cost sensor errors. The retained network contains {a['stations']} distinct monitoring locations, partitioned into eight geographic clients; eight stations are excluded from fitting. We compare an inductive spatial graph model under eight training/aggregation strategies with ordinary kriging and inverse distance weighting, and separately test a temporal LSTM. Across seeds {run['seeds']}, trust-weighted graph FL obtained unseen-station RMSE {tw.rmse_mean:.2f} ± {tw.rmse_sd:.2f} µg/m³, compared with {fa.rmse_mean:.2f} ± {fa.rmse_sd:.2f} for FedAvg, a signed reduction of {gain:.2f}%. The best unseen-station comparator was {LABELS[best.method]} ({best.model}), with RMSE {best.rmse_mean:.2f} µg/m³. These measurements support only the specified simulation and split. We include a corruption sweep, logical communication accounting and an update-based membership-inference probe. No formal privacy or universal accuracy claim is made.

Keywords: federated learning; PM2.5; sensor calibration; spatial prediction; graph neural network; smart cities.

## 1 Project and implementation scope

The assigned project asks whether geographically distributed sensor owners can jointly predict city pollution while retaining raw training readings locally. Its instructor note identifies heterogeneous calibration quality and co-location-based weighting as the main experiment. Earlier submissions described a design without trained results. This implementation supplies executable models, actual predictions, data provenance, a local demonstration dashboard and reproducible measurements. All original submissions remain unchanged.

Each client trains from its own simulated sensor readings. The aggregation function accepts parameter dictionaries and scalar weights, not raw records. The simulator and evaluator necessarily load public reference data to construct the experiment and score outputs; this is not a deployment across independently protected organizations. At mapping time a local client uses its donor histories and returns a prediction. This is a simulation of local inference, not a demonstrated encrypted network protocol. The dashboard is an additional presentation interface beyond the earlier paper's scope.

## 2 Data provenance and suitability

The primary dataset is the India Air Quality Database published by XKDR Forum, sourced here from CPCB's continuous ambient monitoring network. We downloaded {a['source_rows']:,} hourly PM2.5 rows for 2024 from 39 source station entries, with original timestamps, units and coordinates. After excluding insufficient-coverage and co-located entries, {a['retained_observations']:,} valid observations remain at {a['stations']} distinct locations. The compilation is licensed CC BY 4.0 and must be attributed to India Air Quality Database, XKDR Forum, together with CPCB. Source readings are preliminary; this project does not perform regulatory validation. Source: https://airquality.xkdr.org and https://github.com/xKDR/Air-Quality-Database.

The secondary dataset is Song Chen's UCI Beijing Multi-Site Air Quality dataset (DOI 10.24432/C5RK5G), licensed CC BY 4.0. The retained original archive covers 12 reference stations from March 2013 through February 2017, with 420,768 station-hour rows and pollutant/meteorology fields. We run an independent temporal persistence benchmark on it. We do not claim it contains low-cost instruments or that it establishes Delhi-to-Beijing model transfer. Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data.

OpenAQ and sensor.community were considered in the proposal. They are not data sources used in the measured implementation. The selected licensed historical sources are directly downloadable and suitable for a self-contained dataset submission. Weather variables are not used in Delhi models; no unverified meteorology is fabricated. The original UCI archive includes weather fields but the secondary sanity benchmark uses PM2.5 only.

### Cleaning and organization

We align Delhi observations to the 8,784 local hours of leap year 2024. Duplicate station-hour entries are averaged. Values below zero or above 2,000 µg/m³ are removed. Stations require 70% valid coverage. Entries within 50 m of another retained entry are treated as co-located catalog entries; the more complete entry is kept to prevent them straddling training and unseen-station sets. Excluded co-located entries: {a['colocated_catalog_entries_excluded']}. Coverage filtering uses full-year availability, a benchmark-construction choice that should be revisited for prospective deployment.

Only input histories are forward-filled, with a three-hour maximum. We never backfill and never impute the reference target. A sample is excluded when its donor history or reference target is unavailable. PM2.5 inputs and labels are divided by 100; relative coordinates are expressed in kilometers and divided by 50. These fixed scaling constants do not use test statistics. Exact samples, reference data, simulated readings, station ownership and splits are exported under datasets/processed.

## 3 Simulation and experimental design

Client groups are deterministic, approximately balanced latitude-ordered geographic strips. For each client, a virtual sensor is co-located with each retained reference monitor. A seeded subset of clients receives severe synthetic errors. The primary scenario has three of eight corrupted clients (37.5%); the nominal 40% setting is rounded to whole clients.

Virtual reading = max(0, gain × reference + bias + annual_drift × elapsed_year_fraction + Gaussian_noise). Independent dropouts and stuck blocks are then applied. Parameters below are explicit scenario assumptions, not measured Plantower specifications or values attributed to an unverified paper. Relative humidity and physical aerosol growth are not modeled.

{md_table(profiletable)}

Stuck-block starts are sampled uniformly after the first day; a block repeats the preceding reading. Reference missing values propagate into simulated values. Trust scores use only co-located virtual/reference pairs at training stations during the calibration period, avoiding spatial-gradient confounding.

### Chronology and spatial isolation

The first 10% of hours are calibration-only, ending {a['split_boundaries'][0]} IST. Model-training targets occupy 10–70%, validation targets 70–85%, and test targets 85–100%. Training ends {a['split_boundaries'][1]}; testing starts {a['split_boundaries'][2]}. Thus we reserve a separate calibration prefix rather than claiming all of the first 70% is model training. All models use six hours of strictly past history to predict the next hourly target, with samples every three hours. Input history can precede a split boundary, as it would in a rolling prediction setting; future targets never enter earlier fitting.

Eight complete stations, one per geographic client, are held out from training, validation, calibration and donor inputs. They are scored only in the future test period. This is one deterministic spatial holdout, not five-fold spatial cross-validation. Separate known-station and unseen-station tables prevent a forecasting result from being presented as unseen-location mapping evidence. The exact unseen station IDs are {', '.join(a['heldout_stations'])}.

For membership probing, 10% of eligible training-era windows are randomly reserved as nonmembers and removed before optimization. Adjacent histories may overlap; this limitation is explicit. Seed-17 sample counts: {json.dumps(a['sample_counts'])}.

## 4 Models and aggregation

The spatial learner is an inductive star-graph message-passing network. The target has a coordinate but no pollutant reading. Each of up to three other training stations in the same client contributes its six-hour PM2.5 history and relative x/y coordinates. A shared 8→24→16 message network, learned attention, pooled messages and four cyclic time features produce a residual correction to the weighted last donor value. When only two donors are available, the last donor is repeated to obtain a fixed three-message input; this changes its effective multiplicity and is a limitation. Neighbor order is immaterial. Target station IDs are never model inputs.

The temporal learner is a 16-unit LSTM using six hours of the target station's own history plus cyclic time features. It predicts a residual from persistence. It is evaluated only at known stations because it cannot observe a history at a location with no sensor. Its information set is richer than the graph model's target-excluding inputs, so their scores are not a controlled architecture ablation.

Trust confidence c_k = max(exp(−calibration_RMSE_k / 35), 0.05); aggregation weight alpha_k = n_k c_k / sum_j(n_j c_j). Sample-weighted FedAvg sets c_k=1. Equal averaging is implemented separately. No raw record is an argument of the aggregator.

We also implement coordinate-wise median, one-coordinate-tail trimmed mean, Krum with assumed f=1, and an update-consistency comparator (sample count divided by distance from the coordinate median update). The latter is our explicit heuristic; it is not a claimed reproduction of a published reputation algorithm. Krum's theoretical Byzantine condition for the actual corrupted-client count is violated in the primary three-of-eight scenario and the five-of-eight sweep. It remains an empirical comparator only.

Centralized models use the same noisy labels and features with pooled training. They are comparators, not guaranteed accuracy ceilings. For each round they receive eight times the per-client step budget, matching total optimization steps. All neural optimizers are Adam, learning rate 0.002, batch size at most 128, gradient clipping norm 5, {run['rounds']} rounds and {run['local_steps']} local steps per client per round. Adam state is reset for each local training call, including each centralized round. Checkpoints are selected only by clean validation-reference RMSE at known stations; all trainable methods receive this same validation access. No hyperparameter search or test-based tuning is performed. Clean validation feedback is additional trusted information whose deployment cost is not modeled.

IDW uses inverse squared distance. Ordinary kriging uses an exponential spatial covariance and a 0.05 normalized nugget; its range is chosen from 5, 10, 20, 40, 80 km using calibration-period semivariances at seen stations. The last seed's selected range is {run['kriging_range_km']} km. Neither geostatistical method uses the target reading. They operate on the same available client-local donor set and previous-hour observations as the graph task. This is one-hour-ahead spatial prediction, not contemporaneous interpolation using future or target observations. Negative predicted values are clipped to zero consistently in scoring.

## 5 Measured results

Seeds {run['seeds']} vary client corruption identities, generated sensor errors, training/nonmember reservation and optimization randomness. The station holdout is fixed. Results below show mean ± sample standard deviation across seeds. Counts and per-run scores are in results/metrics.csv. Repeated seeds do not establish spatial cross-validation or independent deployment replications.

### Unseen stations in the future test period

{tables['spatial']}

![Unseen-station comparison](../results/figures/comparison_spatial.png)

Trust weighting achieved a signed RMSE reduction of {gain:.2f}% relative to sample-weighted FedAvg. A negative reduction means trust performed worse. The strongest unseen-station result was {LABELS[best.method]} with RMSE {best.rmse_mean:.2f} µg/m³. We do not change baselines or hide outcomes when classical methods or unweighted aggregation are stronger. Input-side calibration errors remain present even when a client contributes less weight, so weighting alone need not correct a map generated from that client's biased donor observations.

### Known stations in the future test period

{tables['test']}

![Known-station comparison](../results/figures/comparison_test.png)

The LSTM and persistence use each target's recent history; the graph and spatial baselines exclude it. Comparisons across these task families therefore require this information-set qualification. Seasonal shift into November/December raises pollution levels relative to much of the model-training period. Calibration is measured in the winter prefix, while drift grows through the year, so initial trust can become stale.

### Robustness and spatial diagnostics

The sweep uses actual corruption fractions 0%, 25%, 37.5% and 62.5%, corresponding to zero, two, three and five of eight clients. Each fraction is evaluated for FedAvg, trust, coordinate median and trimmed mean using the same seeds and budgets. The amount of bias/drift for a severe client is fixed; this experiment varies client share rather than independently sweeping drift magnitude. Individual corruption identities differ across seeds.

![Robustness sweep](../results/figures/robustness.png)

![Measured station errors](../results/figures/spatial_errors.png)

![Validation convergence](../results/figures/convergence.png)

Paired seed-level Wilcoxon tests: {json.dumps(run['significance'])}. Three seeds provide low statistical power; no broad significance claim follows. Per-window independence is not assumed. Error bars summarize simulation seeds rather than dense-grid uncertainty.

## 6 Privacy and communication

The threat model is an honest-but-curious aggregator that sees each client's before/after model parameters. The probe ranks candidate windows by the decrease in their noisy-label squared loss caused by the last individual update. Members are model-training windows; nonmembers are reserved windows from the same era and client. This is an explicit update-based loss-improvement heuristic, motivated by membership-inference literature; it is not a literal reproduction of Yeom's fixed loss-threshold attack. No shadow models, gradient inversion, secure aggregation or differential privacy are implemented.

{md_table(ptable)}

![Privacy probe](../results/figures/privacy_probe.png)

AUC near 0.5 indicates weak ranking for this probe, not proof of privacy. Advantage is maximum TPR−FPR over thresholds; threshold selection on the evaluated attack set is descriptive and optimistic. Candidate histories overlap and the final client model has not necessarily optimized every randomly sampled member in the last round. A stronger adversary or full training transcript may behave differently. The attack evaluates the final-round individual update, while prediction metrics use the validation-selected global checkpoint.

The spatial model contains 1,162 float32 parameters. Across {run['rounds']} rounds and eight clients, logical upload payload is {upload/1e6:.3f} MB; upload plus broadcast payload is {upload*2/1e6:.3f} MB per run, excluding transport overhead. The prepared pooled training-tensor comparator is approximately {rawbytes/1e6:.3f} MB. The FL tensor exchange is {movement:.2f} times that comparator. These are scenario-dependent logical byte counts, not measured WAN traffic or proof of bandwidth savings. We make no automatic communication-reduction claim. Centralized raw upload and FL parameter exchange differ in contents and repetition; preprocessing, metadata, calibration references and inference predictions are outside these counts.

## 7 Demonstration and reproducibility

The FastAPI application serves a local dashboard with model comparisons, a historical model-driven coordinate surface, selectable reference monitors, calibration profiles, convergence, privacy results and dataset/result exports. Every map request runs the saved graph model; it is not a static heatmap. Dense grid locations lack reference labels, so a smooth-looking grid is not validated street-level accuracy. Grid cells are not masked to Delhi's administrative boundary. The latest supported artifacts run entirely offline after source data and dependencies are available.

Run `start.bat` and open http://127.0.0.1:8000. Reproduce training with `python scripts/run_experiments.py`, figures/exports with `python scripts/build_submission.py`, and tests with `python -m pytest`. Runtime versions are captured in requirements-lock.txt and results/run.json. Source and processed-file SHA-256 hashes are in datasets/manifest.json. Tests check aggregation equations, robust-aggregation constraints, chronology, membership isolation, held-out donor exclusion, invariance to held-out truth changes and graph neighbor-order invariance; API tests exercise saved-model inference and export restrictions.

## 8 Conclusions and limits

We completed a reproducible implementation of calibration-weighted federated pollution prediction and evaluated it against centralized, robust and geostatistical comparators. The measured trust-versus-FedAvg unseen-station RMSE difference is {gain:.2f}% under the specified synthetic-error scenario. The data do not support a blanket claim that trust weighting is best, that federation prevents inference, that the method saves bandwidth, or that grid predictions have street-level accuracy.

Limitations include assumed sensor-error parameters without real low-cost/reference validation, a single city/year and station holdout, small client groups, stale calibration, no humidity model, no prospective deployment, limited attack strength, clean-reference checkpoint selection, and sensitivity to donor quality. The secondary Beijing task is a sanity benchmark, not full cross-city validation. A next study should collect paired low-cost/reference data, compare explicit local recalibration with weighting alone, use grouped spatial folds, refresh trust over time and test secure aggregation or privacy mechanisms under an explicit privacy budget.

## 9 Academic and disclosure notes

This document records the original fixed-holdout benchmark. The completed five-fold station cross-validation, input-calibration ablation, relative-error sensitivity and temporal reputation baseline are documented in EXTENDED_EVALUATION.md. The 25-paper comparison is in LITERATURE_REVIEW.md and LITERATURE_COMPARISON.csv. Teacher feedback supplied directly by the user is mapped in REQUIREMENTS.md. The final manuscript includes the supplied student corresponding-author email. Publication status remains not submitted; no acceptance or submission ID is claimed.

Generative AI assisted code development, dataset discovery, report drafting and presentation-interface design. Reported tables and figures are generated programmatically from downloaded observations and executed experiments; sensor errors are generated by the disclosed numerical simulator. No ImageGen or invented experiment results are used. The team should review the implementation and disclosure against course policy before submission.

## References for the implementation

1. India Air Quality Database, XKDR Forum; CPCB CAAQM source observations. https://github.com/xKDR/Air-Quality-Database (CC BY 4.0).
2. Chen, S. Beijing Multi-Site Air Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G (CC BY 4.0).
3. McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. https://proceedings.mlr.press/v54/mcmahan17a.html.
4. Yin et al. Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML 2018. https://proceedings.mlr.press/v80/yin18a.html.
5. Blanchard et al. Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html.
6. Yeom et al. Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. CSF 2018. https://arxiv.org/abs/1709.01604.
'''
    (docs/'IMPLEMENTATION_REPORT.md').write_text(report,encoding='utf-8')
    dataset_readme=f'''# Datasets used by AirTrust

## Attribution and license

Delhi: India Air Quality Database, XKDR Forum, and Central Pollution Control Board (CAAQM network). Compilation license: Creative Commons Attribution 4.0 International. https://github.com/xKDR/Air-Quality-Database and https://airquality.xkdr.org . Only CPCB observations are requested; no AirNow source rows are used.

Beijing: Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. DOI https://doi.org/10.24432/C5RK5G . License: Creative Commons Attribution 4.0 International.

License terms: https://creativecommons.org/licenses/by/4.0/ . Provide credit, a link to the license and indicate modifications when redistributing. No endorsement is implied. The compilation and all derived data are provided without warranty. Source data are not a certified health advisory.

## Exact data used

raw/delhi_pm25_hourly_2024.csv: {a['source_rows']:,} source rows; 39 source station entries; 2024 calendar year, hourly PM2.5, local IST. Columns: station_id, parameter_name, unit, period_start, mean, min, max, n. Original API responses are also retained in raw/delhi_hourly; these are provenance copies, not extra independent observations. Source metadata and coordinates are in raw/delhi_stations.json. Download route: /v1/measurements with city=Delhi, source=cpcb_caaqm, parameter=PM2.5, agg=hourly and seven-day windows to stay below the public demo row cap.

raw/beijing_multi_site_uci.zip: unmodified UCI source bundle. raw/beijing_station_data.zip: nested station CSV archive extracted from that bundle. These are duplicate representations of the same dataset. It has 420,768 reference station-hour rows across 12 stations and 18 source columns. The secondary benchmark uses PM2.5 only. Nested CSVs can be extracted with any ZIP utility.

processed/delhi_reference_cleaned_2024.csv.gz: 8,784 hours × {a['stations']} retained station columns, with missing values preserved. Valid values: {a['retained_observations']:,}. Coverage cutoff 70%; values outside 0–2,000 µg/m³ excluded; entries within 50 m consolidated by keeping the most complete station. Original values otherwise unchanged.

processed/seed_*/client_*_simulated_pm25.csv.gz: virtual client readings (µg/m³), including held-out stations as separately identifiable evaluation-only sensors. Ownership does not imply fitting membership. station_assignments.csv identifies client and held-out status. Those held-out records are excluded by all model training, calibration and donor builders. These files are **synthetic derivatives of real reference observations**, not independent real low-cost sensor measurements. All simulation parameters and exact splits are in simulation_and_split.json.

processed/seed_*/*_samples.npz: exact prepared samples. Fields: x [sample,3 donors,8 features] (six PM2.5 lags divided by 100, relative x and y kilometers divided by 50); own [sample,6] target histories divided by 100 for the temporal task; context [sample,4] hourly/yearly sine/cosine; y noisy target divided by 100; truth unimputed reference divided by 100; t integer hour index; s station index in station_assignments.csv; client zero-based client. train/nonmember sets are disjoint target-window pairs. Models consume x or own, context and noisy y only. Evaluator consumes truth. NPZ arrays intentionally include evaluator truth and target histories even in the spatial split; the graph model never consumes own or truth during fitting/inference.

## Source roles and exclusions

Delhi is the primary federated experiment. Beijing is a separate chronological temporal persistence benchmark; it is not a second federated evaluation. OpenAQ, sensor.community, meteorology services and Kaggle are not used in these results. There are no private individual records.

## Reproduce and verify

Run scripts/download_data.py to retrieve sources, scripts/run_experiments.py to generate results and scripts/build_submission.py to export derived files. Verify every entry in manifest.json with SHA-256. Manifest timestamps record actual download time in UTC, while observation times remain in source local time. The application needs no source API access once the files are present. CC BY attribution applies to source and modified derived data; this README identifies the modifications.
'''
    if (ROOT/'datasets'/'expansion.json').exists():
        expansion=json.loads((ROOT/'datasets'/'expansion.json').read_text())
        dataset_readme+='\n## Expanded contextual observations\n\n'
        for item in expansion['files']:
            dataset_readme+=f"{item['path']}: {item['rows']:,} observed {item['parameter']} rows at {item['stations']} Delhi catalog entries, hourly IST in 2024. CC BY 4.0, India Air Quality Database / XKDR Forum / CPCB CAAQM. Used for contextual monthly trends, not PM2.5 training labels.\n\n"
        dataset_readme+='Delhi totals 966,067 pollutant rows across three parameters; added contextual rows total 648,440. Including the 420,768 Beijing station-hour rows gives 1,386,835 source rows with differing schemas. Duplicated provenance copies and virtual sensor derivatives are excluded from these counts.\n\nThe GitHub source snapshot is bundled/source_data.zip with its SHA-256 in bundled/manifest.json. Run scripts/prepare_local_data.py to restore the exact source tree offline, then scripts/build_submission.py to generate processed samples and submission ZIPs. Extended experiment audits and per-fold metrics are included under results/.\n'
    (ROOT/'datasets'/'README.md').write_text(dataset_readme,encoding='utf-8')
    manifest=json.loads((ROOT/'datasets'/'manifest.json').read_text());manifest['processed_files']=[]
    for p in processed.rglob('*'):
        if p.is_file():manifest['processed_files'].append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest['processing']=dict(retained_stations=a['stations'],valid_reference_observations=a['retained_observations'],seeds=run['seeds'],units='PM2.5 ug/m3; coordinates decimal degrees; timestamps IST',modifications='Filtering, limited causal input filling, synthetic sensor errors, deterministic client/station splits, fixed scaling')
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for zipname,folders in [('datasets_used.zip',['datasets']),('airtrust_project.zip',['airtrust','scripts','web','tests','docs','results'])]:
        with zipfile.ZipFile(ROOT/'submission'/zipname,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for folder in folders:
                for p in (ROOT/folder).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts:
                        z.write(p,Path('airtrust')/p.relative_to(ROOT))
            if zipname=='airtrust_project.zip':
                for name in ['README.md','requirements.txt','requirements-lock.txt','start.bat','reproduce.bat','pytest.ini','LICENSE','.gitignore']:
                    if (ROOT/name).exists():z.write(ROOT/name,Path('airtrust')/name)
                # Sources ship separately; include attribution inside the code package too.
                z.write(ROOT/'datasets'/'README.md','airtrust/datasets/README.md')
                z.write(ROOT/'datasets'/'manifest.json','airtrust/datasets/manifest.json')
        print('Created',zipname,flush=True)
    print('Submission exports complete',flush=True)

if __name__=='__main__':main()
