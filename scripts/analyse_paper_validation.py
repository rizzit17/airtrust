"""Paired seed-level inference conditional on the fixed Delhi benchmark.

Five overlapping station folds are averaged within each seed, never counted as
75 independent replications. All 36 method pairs x 3 metrics form one Holm family.
"""
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t
from airtrust.data import ROOT

def exact_signed_rank(d):
    """Enumerate sign assignments; supports absolute-rank ties and drops zeros."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if not len(d): return 1.0
    ranks = rankdata(np.abs(d), method='average')
    observed = abs(np.dot(np.sign(d), ranks))
    signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    null = abs(signs @ ranks)
    return float(np.mean(null >= observed-1e-12))

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1., np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order]))
    return result

def main():
    out = ROOT/'results'/'paper_validation'
    df = pd.read_csv(out/'metrics.csv')
    spatial = df[df.split == 'spatial']
    assert len(spatial)==675 and spatial.groupby(['seed','method']).size().eq(5).all()
    assert spatial.groupby(['seed','fold'])['n'].nunique().eq(1).all(), 'Methods must score the same target windows'
    seed = spatial.groupby(['seed','method'])[['mae','rmse','r2']].mean().reset_index()
    seed.to_csv(out/'seed_means.csv', index=False)
    rows = []
    methods = sorted(seed.method.unique())
    for metric in ('rmse','mae','r2'):
        wide = seed.pivot(index='seed', columns='method', values=metric)
        assert wide.shape == (15,9) and not wide.isna().any().any()
        for a,b in itertools.combinations(methods,2):
            d = (wide[a]-wide[b]).to_numpy()
            mean, sd = float(d.mean()), float(d.std(ddof=1))
            half = float(t.ppf(.975, len(d)-1)*sd/np.sqrt(len(d)))
            rows.append(dict(metric=metric,method_a=a,method_b=b,n_seeds=len(d),difference_a_minus_b=mean,median_difference=float(np.median(d)),sd_difference=sd,ci95_low=mean-half,ci95_high=mean+half,paired_dz=mean/sd if sd else np.nan,wins_a=int(sum(d<0 if metric!='r2' else d>0)),p_exact_signed_rank=exact_signed_rank(d)))
    result = pd.DataFrame(rows)
    result['p_holm_108'] = holm(result.p_exact_signed_rank)
    result['reject_holm_005'] = result.p_holm_108 < .05
    result.to_csv(out/'paired_comparisons.csv',index=False)
    (out/'statistical_protocol.json').write_text(json.dumps(dict(unit='One seed mean across the same five fixed station folds',pairs=36,metrics=['rmse','mae','r2'],multiplicity_family=108,test='Two-sided exact sign enumeration of Wilcoxon signed ranks, zero differences omitted; requires symmetric paired-difference distribution for signed-rank location interpretation',confidence_intervals='Unadjusted paired t intervals for seed-mean differences, conditional on fixed data, folds and model settings; not simultaneous or population-station intervals',null_interpretation='Failure to reject is not equivalence or proof of no effect; no equivalence margin was prespecified',scope='15 stochastic replications of the same Delhi year; no additional city, station or field measurement'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(out/'summary.csv')
    s=summary[summary.split=='spatial'].sort_values('rmse_mean')
    names={'centralized':'Centralised GNN','fedavg':'FedAvg GNN','trust':'Trust GNN','median':'Median GNN','trimmed':'Trimmed mean GNN','krum':'Krum GNN','reputation':'Update consistency GNN','idw':'IDW','kriging':'Ordinary kriging'}
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    ax.barh([names[m] for m in s.method],s.rmse_mean,xerr=s.rmse_sd,color=['#277357' if m=='trust' else '#aabfbb' for m in s.method],capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel='RMSE (µg/m³), mean ± fold-seed SD',title='Unseen-station mapping: five station folds × fifteen seeds')
    ax.grid(axis='x',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT/'results'/'figures'/'spatial_cv_extended.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(result[(result.metric=='rmse') & (result.method_a=='fedavg') & (result.method_b=='trust')].to_string(index=False))

if __name__ == '__main__': main()
