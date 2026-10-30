"""Check the exact paired test and multiplicity adjustment used in the paper."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

path=Path(__file__).resolve().parents[1]/'scripts'/'analyse_paper_validation.py'
spec=importlib.util.spec_from_file_location('paper_statistics',path)
stats=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

def test_exact_signed_rank_matches_independent_scipy_calculation():
    for differences in ([1,2,3,4,5], [-1,2,-3,4,5], [3,-7,2,-9,6,11]):
        assert np.isclose(stats.exact_signed_rank(differences),wilcoxon(differences,method='exact').pvalue)

def test_exact_signed_rank_handles_ties_zeros_and_direction():
    assert stats.exact_signed_rank([0,0,0])==1
    assert stats.exact_signed_rank([1,1,1,1])==.125
    d=np.array([0,1,-2,2,3,-4])
    assert stats.exact_signed_rank(d)==stats.exact_signed_rank(-d)
    assert stats.exact_signed_rank(d)==stats.exact_signed_rank(d[d!=0])

def test_holm_preserves_input_order_and_stepdown_monotonicity():
    assert np.allclose(stats.holm([.04,.01,.03]),[.06,.03,.06])
    assert np.allclose(stats.holm([1,1,1]),[1,1,1])
    assert np.allclose(stats.holm([0,.5,.01]),[0,.5,.02])
