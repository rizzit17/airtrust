# Second-level manuscript revision

The revised paper is **A Reproducible Delhi Benchmark of Calibration and Aggregation in Federated PM2.5 Mapping**. It retains the original Word template and reports a benchmark result rather than promising superior trust aggregation or formal privacy.

## Response to the six comments

| Comment | Change and evidence |
|---|---|
| Rewrite the prose | Narrative sections rewritten in direct English; numerical tables and references retained and checked. Filler language and proposal-era claims removed from the rewritten text. |
| Reframe the title | Title, abstract, contributions, discussion and conclusion centre the reproducible benchmark, calibration-versus-aggregation distinction and measured outcome. |
| Drop or earn privacy-preserving | Removed the claim from the title. The manuscript consistently describes update sharing, single-machine execution and a limited membership probe. No differential privacy, secure aggregation or general privacy guarantee is claimed. |
| More seeds and paired tests across methods | Expanded spatial validation from 3 to 15 seeds, retaining the same five station folds and training settings. Nine spatial methods yield 675 fold-seed scores. Five folds are averaged within each seed before paired analysis. All 36 method pairs for all three metrics are exported, with Holm correction across 108 tests. Conditional paired intervals and effects are reported; nonsignificance is not equivalence. |
| Validate the sensor model or label it illustrative | Chose the reviewer's explicitly permitted illustrative option. Both the bias/gain/drift stress test and Kanpur-magnitude sensitivity are labelled synthetic assumptions, not field-fitted or hardware-validated models. No new paired hardware measurements were obtained. |
| Separate LSTM from spatial mapping | Abstract, methods, captions and conclusions explicitly distinguish own-history known-station forecasting from donor-based unseen-station mapping. |

## What was trained

The original three seeds (17, 29, 43) are preserved. Twelve new seeds (59, 71, 89, 101, 127, 149, 173, 197, 223, 251, 277, 307) repeat five station folds and seven neural spatial methods. This adds **420 graph-model fits** with the existing architecture and hyperparameters. IDW and kriging are recomputed for the new seeds as well.

The original fixed-holdout comparison, input-calibration ablation, temporal LSTM, sensor-sensitivity experiment and privacy probe remain three-seed experiments. The hosted dashboard retains its original three-seed checkpoints. The expanded statistical evidence is an offline research evaluation, not a change to the live model or sensor-control behaviour.

Additional seeds create reproducible virtual derivatives of the same observations. They do not add real source rows, stations or hardware measurements to the dataset count.

## Expanded result

Across 75 fold-seed scores, trust GNN RMSE is **96.81 ± 11.10 µg/m³**, compared with **96.08 ± 11.01** for FedAvg. The paired seed-mean difference (trust minus FedAvg) is **+0.73 µg/m³**, with an unadjusted paired 95% interval **+0.17 to +1.29**. The exact signed-rank probability is **0.0125** before correction and **0.6475** after Holm correction across the 108-test family. This shows no trust advantage and does not establish equivalence.

The adjusted RMSE comparisons favour FedAvg over IDW, ordinary kriging and trimmed-mean aggregation (adjusted probabilities 0.0066, 0.0066 and 0.0403). The original calibration ablation remains separate: input correction reduces trust RMSE by 21.27% and helps FedAvg similarly. These results are conditional on the specified synthetic-fault benchmark.

## Reproduce

From the repository root, after restoring the public datasets and installing requirements:

```bash
python scripts/prepare_local_data.py
python scripts/run_paper_validation.py
python scripts/analyse_paper_validation.py
python -m pytest tests -q
```

The training script resumes completed per-run metric files and reuses the original three-seed CV files. `results/spatial_cv` remains unchanged. New outputs live in `results/paper_validation`:

- `metrics_<seed>_<fold>_<method>.csv`: individual measured outputs.
- `audit_<seed>_<fold>.json`: station exclusions, sensor assumptions and sample counts.
- `summary.csv`: descriptive means and SDs across 75 fold-seed scores per spatial method.
- `seed_means.csv`: the 15 paired stochastic units, averaging five folds within each seed.
- `paired_comparisons.csv`: all 108 comparisons, conditional mean-difference intervals, paired effects, exact signed-rank probabilities and Holm-adjusted probabilities.
- `protocol.json` and `statistical_protocol.json`: settings, assumptions and limits.

The exact signed-rank calculation enumerates sign assignments, averages tied ranks and omits zero differences. Interpreting the test as a location comparison assumes symmetric paired differences. The implementation is checked against [SciPy's signed-rank calculation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) on tie-free examples and independently tested for ties, zeros and direction symmetry. Holm correction uses the step-down adjustment described in the [statsmodels implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html).

The 95% paired t intervals are unadjusted and conditional on the same Delhi year, station folds and scenario. They are not simultaneous intervals, independent-station confidence intervals or evidence about a new physical deployment. No equivalence margin was prespecified.

## What remains outside this paper's claims

Real co-located low-cost/reference field validation, other cities and years, physical federation, secure aggregation and differential privacy remain future research. None is presented as completed. The current paper does not need a new model architecture or a privacy model to support its deliberately limited benchmark claim.
