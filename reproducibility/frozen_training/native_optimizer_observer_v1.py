"""Observe an unchanged native optimizer_step, including warmup zero-LR groups.

Tensor math, clipping, GradScaler, optimizer and native EMA semantics are retained.
A Python fused-step hook is a call count only, never an effective-update count.
"""
import torch


def assert_finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():assert_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:assert_finite(v)


def state_step(optimizer,parameter):
    value=optimizer.state.get(parameter,{}).get('step',0)
    return float(value.item()) if isinstance(value,torch.Tensor) else float(value)


def observe_native_optimizer_step(trainer,native_step):
    """Temporarily observe native unscale/clip returns and tracked actual states.

    One gradient-bearing representative per native parameter group is tracked.
    The step must advance each tracked AdamW counter and change at least one
    tracked parameter with nonzero LR. Initial zero-LR backbone is permitted.
    Overflow is recorded only for nonfinite unscaled gradients, falling scaler,
    unchanged tracked steps/parameters. EMA still follows the native call count.
    """
    tracked=[]
    for group in trainer.optimizer.param_groups:
        candidates=[p for p in group['params'] if p.requires_grad and p.grad is not None]
        assert candidates,'Native AdamW group has no gradient-bearing parameter'
        p=candidates[0]
        tracked.append(dict(parameter=p,before=p.detach().clone(),step=state_step(trainer.optimizer,p),
            group_kind=group.get('param_group'),lr=float(group['lr'])))
    assert tracked and any(t['lr']>0 for t in tracked),'All native parameter-group LRs are zero; review the recipe'
    old_scale=float(trainer.scaler.get_scale());ema_before=int(trainer.ema.updates)
    original_unscale=trainer.scaler.unscale_;original_clip=torch.nn.utils.clip_grad_norm_
    calls=dict(unscale=0,clip=0)
    observed={}
    def unscale(optimizer):
        assert optimizer is trainer.optimizer
        result=original_unscale(optimizer);calls['unscale']+=1
        grads=[p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert grads
        observed['unscaled_gradient_tensors']=len(grads)
        observed['unscaled_gradients_finite']=all(bool(torch.isfinite(g).all()) for g in grads)
        return result
    def clip(*args,**kwargs):
        result=original_clip(*args,**kwargs);calls['clip']+=1
        observed['native_gradient_norm_before_clip']=float(result) if torch.isfinite(result) else None
        observed['native_gradient_norm_finite']=bool(torch.isfinite(result))
        if observed['unscaled_gradients_finite']:assert torch.isfinite(result),'Finite gradients produced a nonfinite native clip norm'
        return result
    trainer.scaler.unscale_=unscale;torch.nn.utils.clip_grad_norm_=clip
    try:native_step()
    finally:
        trainer.scaler.unscale_=original_unscale;torch.nn.utils.clip_grad_norm_=original_clip
    assert calls==dict(unscale=1,clip=1)
    new_scale=float(trainer.scaler.get_scale())
    rows=[dict(group_kind=t['group_kind'],lr=t['lr'],state_step_before=t['step'],state_step_after=state_step(trainer.optimizer,t['parameter']),
        parameter_changed=not torch.equal(t['before'],t['parameter'])) for t in tracked]
    advanced=all(t['state_step_after']==t['state_step_before']+1 for t in rows)
    changed_active=any(t['parameter_changed'] and t['lr']>0 for t in rows)
    effective=observed['unscaled_gradients_finite'] and new_scale>=old_scale and advanced and changed_active
    if not effective:
        assert not observed['unscaled_gradients_finite'] and new_scale<old_scale
        assert all(t['state_step_after']==t['state_step_before'] and not t['parameter_changed'] for t in rows)
    assert trainer.ema.updates==ema_before+1,'Preserve native EMA advancement even after AMP overflow'
    assert_finite(trainer.model.state_dict());assert_finite(trainer.ema.ema.state_dict());assert_finite(trainer.optimizer.state_dict())
    return dict(**observed,tracked_groups=rows,scale_before=old_scale,scale_after=new_scale,
        effective_optimizer_step=effective,amp_overflow_skip=not effective,
        native_ema_updates_before=ema_before,native_ema_updates_after=int(trainer.ema.updates),
        hook_is_effective_update_proof=False,zero_lr_backbone_permitted=True,
        scope='One representative per gradient-bearing parameter group; not a proof that every model parameter changes.')
