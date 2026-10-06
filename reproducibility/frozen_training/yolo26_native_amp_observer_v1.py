"""Require the unchanged modern native AMP comparison to actually pass.

This helper observes returns only. It does not replace tensor computations,
device queries, models, or the native0.5 tolerance.
"""
import hashlib,inspect,sys
from pathlib import Path
EXPECTED_NATIVE_SHA='4c694935c327ebd07250d12fc2a2694bb50a4ccad793398adfd8dafdeca9c8c6'
def require_observed_native_amp_check(native_check,model,evidence):
    source=Path(inspect.getfile(native_check)).resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==EXPECTED_NATIVE_SHA
    assert native_check.__name__=='check_amp'
    previous=sys.getprofile();assert previous is None
    observations=[]
    def profile(frame,event,result):
        if event=='return' and frame.f_code.co_filename==native_check.__code__.co_filename and frame.f_code.co_name=='amp_allclose':
            observations.append(dict(returned_exactly_true=result is True))
    result=None
    try:
        sys.setprofile(profile);result=native_check(model)
    finally:
        sys.setprofile(previous)
        evidence.update(native_check_source_sha256=EXPECTED_NATIVE_SHA,native_returned_true=result is True,
            observed_native_inner_comparisons=observations,observed_comparison_passed=observations==[{'returned_exactly_true':True}],
            observation_method='Profile the existing amp_allclose return only. Native tensor math, device, detector and tolerance0.5 remain unchanged. Reject True when the comparison was skipped.')
    assert result is True and observations==[{'returned_exactly_true':True}], 'Native AMP comparison was skipped or failed'
    return result
