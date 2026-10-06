"""Independent native short3 restore; never enters the research train loop.

CPU checks restored model/EMA/optimizer and preserved scaler payload. A CUDA
probe also checks the actual native scaler and one finite backward, with no
optimizer/EMA update, checkpoint save or research CSV/input history change.
"""
from pathlib import Path
import argparse,copy,json,os,sys,time,traceback
from native11_h50_guard_v1 import ROOT,admit,atomic,load,sha,tensor_state_sha,step_deadline

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['restore_CPU','restore_GPU'],required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    protocol,release=admit(a.protocol,a.release,a.stage,__file__)
    short=ROOT/protocol['paths']['short3'];history_path=short/'run_manifest.json';history=load(history_path)
    assert history['status']=='completed_bounded_h50_short3' and history['passed'] and history['completed_training_epochs']==3
    assert history['protocol_SHA256']==sha(a.protocol)
    assert release['actual_short_manifest_SHA256']==sha(history_path)
    source=Path(history['retained_checkpoint']);assert sha(source)==history['retained_checkpoint_SHA256']
    protected={history_path:sha(history_path),source:sha(source),short/'native/results.csv':history['results_csv_SHA256'],
               short/'input_sources.jsonl':history['input_sources_SHA256'],short/'optimizer_observations.jsonl':history['optimizer_observations_SHA256']}
    for f,digest in protected.items():assert sha(f)==digest
    out=ROOT/protocol['paths'][a.stage];assert not out.exists();out.mkdir(parents=True)
    started=time.time();report=dict(status='starting',stage=a.stage,source_SHA256=sha(__file__),protocol_SHA256=sha(a.protocol),
        release_SHA256=sha(a.release),actual_short_manifest_SHA256=sha(history_path),retained_checkpoint_SHA256=sha(source),
        started_unix=started,paper_accuracy_claim=False,formal_training_started=False,optimizer_steps=0,EMA_updates_added=0,
        research_train_loop_called=False,trained_checkpoint_written=False,automatic_next_job_allowed=False)
    target=out/'restore_report.json';atomic(target,report)
    threads=2 if a.stage=='restore_CPU' else 4
    os.environ.update(OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='2',
        YOLO_CONFIG_DIR=str(out/'config'),NO_ALBUMENTATIONS_UPDATE='1')
    sys.path.insert(0,str(ROOT/'engineering_code'))
    try:
        import torch,ultralytics
        from model_and_trainer import MatchedNativeTrainer as DetectionTrainer
        from native_optimizer_observer_v1 import assert_finite
        from ultralytics.engine import trainer as native_module
        torch.set_num_threads(threads);assert ultralytics.__version__=='8.4.170' and torch.__version__=='2.1.2+cu121'
        for relative,digest in protocol['native_source_SHA256'].items():assert sha(Path(ultralytics.__file__).parent/relative)==digest
        saved=torch.load(source,map_location='cpu',weights_only=False)
        assert saved['epoch']==2 and saved['train_args']['epochs']==50 and saved['optimizer'] is not None
        assert saved['updates']==history['native_optimizer_attempts'] and isinstance(saved['scaler'],dict)
        assert_finite(saved['optimizer']);assert_finite(saved['scaler']);assert_finite(saved['ema'].state_dict())
        expected_model=tensor_state_sha(saved['ema'].float());expected_optimizer=copy.deepcopy(saved['optimizer'])
        expected_scaler=copy.deepcopy(saved['scaler']);del saved
        native=out/'native';settings=dict(protocol['recipe'],model=str(source),resume=str(source),
            data=str(ROOT/protocol['data_yaml']),save_dir=str(native),project=str(out),name='native',exist_ok=True,plots=False,
            device='cpu' if a.stage=='restore_CPU' else 0,workers=0 if a.stage=='restore_CPU' else protocol['recipe']['workers'])
        class RestoredNative11(DetectionTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                assert weights is not None and weights.model[-1].nc==10
                model=super().get_model(cfg,weights,verbose)
                assert not model.end2end and model.model[-1].reg_max==16
                model.model[-1].max_det=1000;return model
            def optimizer_step(self):raise AssertionError('Restore probe must not update optimizer or EMA')
            def save_model(self):raise AssertionError('Restore probe must not save a learned checkpoint')
        original_amp=native_module.check_amp
        if a.stage=='restore_GPU' and settings['amp']:
            from yolo26_native_amp_observer_v1 import require_observed_native_amp_check
            def observed_amp(model):
                evidence={}
                try:return require_observed_native_amp_check(original_amp,model,evidence)
                finally:report['native_AMP_check_observation']=evidence;atomic(target,report)
            native_module.check_amp=observed_amp
        tr=RestoredNative11(overrides=settings);tr._setup_train();step_deadline(release,started)
        assert tr.start_epoch==3 and tr.scheduler.last_epoch==2 and tr.epochs==50
        assert tr.ema.updates==history['native_optimizer_attempts']
        assert tensor_state_sha(tr.model)==tensor_state_sha(tr.ema.ema)==expected_model
        assert len(tr.train_loader.dataset)==6471 and len(tr.test_loader.dataset)==548
        actual=tr.optimizer.state_dict();assert set(actual['state'])==set(expected_optimizer['state'])
        # The frozen native restore explicitly replaces fused by runtime device.
        for wanted,group in zip(expected_optimizer['param_groups'],actual['param_groups']):wanted['fused']=group.get('fused')
        assert actual['param_groups']==expected_optimizer['param_groups']
        for k,original in expected_optimizer['state'].items():
            assert set(actual['state'][k])==set(original)
            for key,value in original.items():
                restored=actual['state'][k][key]
                assert torch.equal(restored.cpu(),value.cpu()) if isinstance(value,torch.Tensor) else restored==value
        if a.stage=='restore_CPU':
            assert tr.device.type=='cpu' and not torch.cuda.is_available() and not tr.amp
            if expected_scaler:
                assert not tr.scaler.is_enabled() and tr.scaler.state_dict()=={}
                scaler_scope='Saved AMP scaler payload retained and finite; actual native CPU scaler disabled. CUDA restore is required separately.'
            else:
                assert tr.scaler.state_dict()==expected_scaler
                scaler_scope='Native FP32 empty scaler state exactly restored on CPU.'
            report.update(GPU_used=False,backward_calls=0,actual_native_scaler_payload_equal=not bool(expected_scaler))
        else:
            assert tr.device.type=='cuda' and bool(tr.amp)==protocol['recipe']['amp']
            assert tr.scaler.state_dict()==expected_scaler
            ema_before=tensor_state_sha(tr.ema.ema);updates=tr.ema.updates
            tr.epoch=3;tr.model.train();tr.optimizer.zero_grad(set_to_none=True)
            batch=next(iter(tr.train_loader));assert batch['img'].dtype==torch.uint8
            dense=tr.preprocess_batch(batch);torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda',dtype=torch.float16,enabled=tr.amp):loss,items=tr.model(dense)
            assert_finite(loss);assert_finite(items)
            tr.scaler.scale(loss.sum()).backward();tr.scaler.unscale_(tr.optimizer)
            gradients=[v.grad for v in tr.model.parameters() if v.grad is not None]
            assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
            assert tr.ema.updates==updates and tensor_state_sha(tr.ema.ema)==ema_before
            unchanged=tr.optimizer.state_dict()
            assert unchanged['param_groups']==actual['param_groups']
            for k,state in actual['state'].items():
                for key,value in state.items():
                    now=unchanged['state'][k][key]
                    assert torch.equal(now,value) if isinstance(value,torch.Tensor) else now==value
            assert tr.scaler.state_dict()==expected_scaler
            tr.optimizer.zero_grad(set_to_none=True);step_deadline(release,started)
            scaler_scope='Actual native CUDA scaler state exactly restored; finite unscaled gradients in one discarded engineering backward.'
            report.update(GPU_used=True,backward_calls=1,actual_native_scaler_payload_equal=True,
                finite_backward_and_unscaled_gradients=True,gradient_tensors=len(gradients),
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for f,digest in protected.items():assert sha(f)==digest
        report.update(status='completed_restore_probe',passed=True,completed_unix=time.time(),start_epoch=3,scheduler_last_epoch=2,
            restored_EMA_updates=tr.ema.updates,restored_model_and_EMA_tensor_state_SHA256=expected_model,
            all_model_EMA_optimizer_payload_checks_passed=True,scaler_scope=scaler_scope,
            saved_scaler_payload=expected_scaler,all_short3_research_artifacts_unchanged=True,
            native_half_EMA_and_optimizer_rounding_preserved=True,restore_is_not_bitwise_uninterrupted_training=True)
        atomic(target,report)
    except Exception:
        report.update(status='failed_stop_no_retry',passed=False,error=traceback.format_exc(),failed_unix=time.time());atomic(target,report);raise
    finally:
        if 'native_module' in locals():native_module.check_amp=original_amp
if __name__=='__main__':main()
