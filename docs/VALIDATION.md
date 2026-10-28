# Validation evidence

The measured suite completed with three seeds, 20 rounds and 12 steps per client per round. It trained 33 primary neural configurations and 36 additional robustness configurations. Per-run metrics, predictions, checkpoints and training histories are retained.

The final automated checks cover aggregation equations, robust-aggregation conditions, temporal split ordering, disjoint membership targets, complete held-out station exclusion, invariance to changes in held-out reference truth, graph neighbor-order invariance, distinct station coordinates, original unimputed evaluation targets, UCI hourly-grid integrity, exact checkpoint-to-export prediction consistency, API availability, model-driven maps, input bounds and export restrictions.

The local browser was used to verify the overview, model-driven map, monitor details, comparison split selector, client trust/convergence, privacy tables and dataset export links. Source and processed dataset checksums and both ZIP integrity checks passed. SHA-256 verification covers 104 source and derived files.

The local Git repository records the implementation source. Downloaded datasets and submission ZIPs are excluded from Git; their integrity is recorded separately in the dataset manifest. Original supplied course files remain unchanged.
