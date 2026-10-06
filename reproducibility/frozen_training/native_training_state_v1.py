"""Additional actual-state retention and fail-fast native serialization checks.

The native writer remains intact. Automatic continuation is never provided.
"""
from pathlib import Path
import hashlib,json,time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def to_cpu(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [to_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(to_cpu(v) for v in value)
    return value

def assert_finite(value):
    import torch
    if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():assert_finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:assert_finite(v)

def assert_native_serialization_preconditions(trainer):
    """Reject before native EMA repair/FP16 saturation can obscure failures.

    Fixed8.4.170 writer half-converts EMA; optimizer conversion retains step and
    exp_avg_sq inFP32, converting other FP32 state tensors. This check does not
    change weights, scaler, native writer or rounding and is not a learning test.
    """
    import torch
    assert_finite(trainer.model.state_dict())
    assert_finite(trainer.ema.ema.state_dict())
    assert_finite(trainer.optimizer.state_dict())
    for v in trainer.ema.ema.state_dict().values():
        if isinstance(v,torch.Tensor) and v.is_floating_point():assert_finite(v.half())
    for row in trainer.optimizer.state.values():
        for k,v in row.items():
            if k not in {'step','exp_avg_sq'} and isinstance(v,torch.Tensor) and v.dtype==torch.float32:assert_finite(v.half())

def capture_training_state(trainer,destination,native_checkpoint,identities,engineering_cpu_fixture=False):
    """Keep train model, FP32 EMA/optimizer, scaler and available scheduler.

    Unsaved accumulated grads, RNG, sampler/worker/iterator state preclude a
    claim of bitwise uninterrupted continuation. No continuation is launched.
    """
    import torch
    destination=Path(destination);assert not destination.exists()
    if engineering_cpu_fixture:
        assert trainer.device.type=='cpu' and not torch.cuda.is_available()
    else:
        assert trainer.device.type=='cuda' and trainer.amp and trainer.scaler.is_enabled()
        assert trainer.scaler.state_dict()
    criterion=getattr(trainer.model,'criterion',None)
    progress={k:getattr(criterion,k) for k in ['updates','o2m','o2o','total','final_o2m'] if hasattr(criterion,k)}
    state=dict(kind='native_checkpoint_cpu_fixture' if engineering_cpu_fixture else 'native_detector_full_precision_training_state',
        epoch=int(trainer.epoch),horizon_epochs=int(trainer.args.epochs),ema_updates=int(trainer.ema.updates),
        identities=dict(identities),native_checkpoint_sha256=sha(native_checkpoint),
        model_state_dict=to_cpu(trainer.model.state_dict()),ema_state_dict=to_cpu(trainer.ema.ema.state_dict()),
        optimizer_state_dict=to_cpu(trainer.optimizer.state_dict()),scaler_state_dict=dict(trainer.scaler.state_dict()),
        scheduler_state_dict=to_cpu(trainer.scheduler.state_dict()) if getattr(trainer,'scheduler',None) else None,
        criterion_progress=progress,actual_train_settings=dict(vars(trainer.args)),
        scope='Supplemental actual state evidence only; no automatic resume/release. Native serialization remains separate. Accumulated grads and worker/RNG/iterator/sampler states are not retained; criterion progress is recorded, not automatically restored. Bitwise uninterrupted equivalence is not claimed.',created_unix=time.time())
    for k in ['model_state_dict','ema_state_dict','optimizer_state_dict']:assert_finite(state[k])
    tmp=destination.with_suffix('.pt.part');assert not tmp.exists()
    torch.save(state,tmp);tmp.replace(destination)
    return dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),epoch=state['epoch'],
        ema_updates=state['ema_updates'],native_checkpoint_sha256=state['native_checkpoint_sha256'],
        scaler_state=state['scaler_state_dict'],criterion_progress=progress,engineering_cpu_fixture=engineering_cpu_fixture)

def restore_cpu_fixture(trainer,state,identities):
    """CPU serialization fixture only, intentionally unavailable for GPU resume."""
    import torch
    assert trainer.device.type=='cpu' and not torch.cuda.is_available()
    assert state['kind']=='native_checkpoint_cpu_fixture' and state['identities']==identities
    assert state['horizon_epochs']==trainer.args.epochs and not state['criterion_progress']
    for k in ['model_state_dict','ema_state_dict','optimizer_state_dict']:assert_finite(state[k])
    trainer.model.load_state_dict(state['model_state_dict'],strict=True)
    trainer.ema.ema.load_state_dict(state['ema_state_dict'],strict=True);trainer.ema.updates=state['ema_updates']
    trainer.optimizer.load_state_dict(state['optimizer_state_dict'])
    trainer.scaler.load_state_dict(state['scaler_state_dict'])
    if state['scheduler_state_dict'] is not None:
        trainer.scheduler.load_state_dict(state['scheduler_state_dict'])
    assert trainer.scaler.state_dict()==state['scaler_state_dict']
