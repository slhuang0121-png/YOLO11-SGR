"""Native setup/capacity/short/final helpers; all execution requires admission."""
from pathlib import Path
import copy,csv,json,math,os,shutil,sys,time,traceback
from native11_h50_guard_v1 import ROOT,admit,atomic,load,sha,tensor_state_sha,resources,step_deadline,planning_gate

def run(stage,entry,protocol_path,release_path):
    protocol,release=admit(protocol_path,release_path,stage,entry)
    assert stage in ('setup_CPU','capacity_GPU','short3','final50')
    base=ROOT/protocol['paths'][stage];assert not base.exists();base.mkdir(parents=True)
    started=time.time();report_path=base/'run_manifest.json'
    report=dict(status='starting',stage=stage,started_unix=started,entry_source_SHA256=sha(entry),
        protocol_SHA256=sha(protocol_path),release_SHA256=sha(release_path),schedule_horizon=50,
        formal_training_released=stage=='final50',paper_accuracy_claim=False,automatic_next_job_allowed=False,
        system_comparison_not_strict_architecture_causality=True,variant=protocol['variant'],
        new_C0_peer_required=True,legacy11_is_system_reference=True,epochs=[],completed_training_epochs=0,
        native_optimizer_attempts=0,effective_optimizer_updates=0,AMP_overflow_skips=0,batches=0)
    def save(): atomic(report_path,report)
    threads=2 if stage=='setup_CPU' else 4
    save();os.environ.update(YOLO_CONFIG_DIR=str(base/'config'),OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),
                             OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
    sys.path.insert(0,str(ROOT/'engineering_code'))
    try:
        import numpy as np,torch,ultralytics
        from model_and_trainer import MatchedNativeTrainer as DetectionTrainer
        from ultralytics.nn.tasks import load_checkpoint
        from native_optimizer_observer_v1 import observe_native_optimizer_step,assert_finite
        from native_training_state_v1 import assert_native_serialization_preconditions
        torch.set_num_threads(threads);assert ultralytics.__version__=='8.4.170' and torch.__version__=='2.1.2+cu121'
        code=Path(ultralytics.__file__).parent
        for relative,digest in protocol['native_source_SHA256'].items():assert sha(code/relative)==digest,relative
        init_dir=ROOT/protocol['paths']['initialization'];proof=load(init_dir/'initialization_proof.json')
        initial=Path(proof['checkpoint']);assert proof['passed'] and proof['VisDrone_training_epochs']==0
        assert proof['protocol_SHA256']==sha(protocol_path) and sha(initial)==proof['checkpoint_SHA256']
        initial_state=proof['tensor_state_SHA256'];native=base/'native'
        history=None
        if stage=='final50':
            short=ROOT/protocol['paths']['short3'];history=load(short/'run_manifest.json')
            assert history['status']=='completed_bounded_h50_short3' and history['completed_training_epochs']==3
            assert history['protocol_SHA256']==sha(protocol_path)
            assert planning_gate(history,time.time(),reserve=release['source_seal_reserve_seconds'])['eligible_for_separate_final_review']
            assert release['actual_short_manifest_SHA256']==sha(short/'run_manifest.json')
            assert release['source_seal_reserve_seconds']>=3600
            assert set(release['reviewed_actual_restore_report_SHA256'])=={'restore_CPU','restore_GPU'}
            for probe_stage in ('restore_CPU','restore_GPU'):
                probe_path=ROOT/protocol['paths'][probe_stage]/'restore_report.json';probe=load(probe_path)
                assert sha(probe_path)==release['reviewed_actual_restore_report_SHA256'][probe_stage]
                assert release['evidence_SHA256'][str(probe_path.relative_to(ROOT))]==sha(probe_path)
                assert probe['status']=='completed_restore_probe' and probe['passed']
                assert probe['stage']==probe_stage and probe['protocol_SHA256']==sha(protocol_path)
                assert probe['actual_short_manifest_SHA256']==sha(short/'run_manifest.json')
                assert probe['retained_checkpoint_SHA256']==history['retained_checkpoint_SHA256']
                assert probe['source_SHA256']==sha(Path(entry).parent/'native11_h50_restore_probe_v1.py')
                assert probe['all_model_EMA_optimizer_payload_checks_passed'] and probe['all_short3_research_artifacts_unchanged']
                assert probe['start_epoch']==3 and probe['scheduler_last_epoch']==2 and probe['optimizer_steps']==probe['EMA_updates_added']==0
                if probe_stage=='restore_GPU':assert probe['finite_backward_and_unscaled_gradients'] and probe['actual_native_scaler_payload_equal']
            retained=Path(history['retained_checkpoint']);assert sha(retained)==history['retained_checkpoint_SHA256']
            native.mkdir();resume=native/'weights/resume_epoch3_native.pt';resume.parent.mkdir()
            shutil.copyfile(retained,resume)
            for filename in ('results.csv','input_sources.jsonl','optimizer_observations.jsonl','prepared_first_batch_SHA.jsonl','bounded_assignment_diagnostics.jsonl'):
                source=short/('native/results.csv' if filename=='results.csv' else filename)
                target=native/'results.csv' if filename=='results.csv' else base/filename
                assert not target.exists();shutil.copyfile(source,target)
            # Inherit the actual unstripped selected best. A stripped short best
            # loses epoch/EMA if none of the following 47 epochs improves it.
            previous_best=short/'native/weights/selected_best_before_final_strip.pt'
            assert sha(previous_best)==history['selected_best_before_strip_SHA256']
            raw_best=torch.load(previous_best,map_location='cpu',weights_only=False)
            stripped_best=torch.load(short/'native/weights/best.pt',map_location='cpu',weights_only=False)
            assert raw_best['epoch']+1==history['native_best_epoch'] and raw_best['ema'] is not None
            assert stripped_best['epoch']==-1 and stripped_best['ema'] is None
            assert sha(short/'native/weights/best.pt')==history['best_checkpoint_SHA256']
            assert tensor_state_sha(raw_best['ema'].float())==tensor_state_sha(stripped_best['model'].float())
            del raw_best,stripped_best
            shutil.copyfile(previous_best,native/'weights/best.pt')
            ck=torch.load(resume,map_location='cpu',weights_only=False)
            assert ck['epoch']==2 and ck['train_args']['epochs']==50 and ck['optimizer'] is not None
            assert ck['updates']==history['native_optimizer_attempts']
            assert isinstance(ck['scaler'],dict)
            expected_restored_state=tensor_state_sha(ck['ema'].float())
            expected_optimizer=copy.deepcopy(ck['optimizer']);expected_scaler=copy.deepcopy(ck['scaler']);del ck
            report.update(epochs=history['epochs'].copy(),completed_training_epochs=3,batches=history['batches'],
                native_optimizer_attempts=history['native_optimizer_attempts'],effective_optimizer_updates=history['effective_optimizer_updates'],
                AMP_overflow_skips=history['AMP_overflow_skips'],initial_tensor_state_SHA256=initial_state,
                source_short_manifest_SHA256=sha(short/'run_manifest.json'),retained_source_checkpoint_SHA256=sha(resume),
                native_best_epoch=history['native_best_epoch'],start_completed_epoch=3,additional_training_epochs=47,
                resume_limit='Native half EMA/optimizer rounding, RNG/loader reseed and abandoned accumulated gradients retained; not bitwise uninterrupted training.')
            model_path=resume
        else:model_path=initial
        settings=dict(protocol['recipe'],model=str(model_path),data=str(ROOT/protocol['data_yaml']),
                      save_dir=str(native),project=str(base),name='native',exist_ok=True,plots=False)
        if stage=='setup_CPU':settings.update(device='cpu',workers=0,amp=False)
        else:settings.update(device=0)
        settings['resume']=str(model_path) if stage=='final50' else False
        inputs=base/'input_sources.jsonl';observations=base/'optimizer_observations.jsonl'
        if stage!='final50':inputs.touch(exist_ok=False);observations.touch(exist_ok=False)
        report['initial_tensor_state_SHA256']=initial_state
        report['best_written_checkpoint_SHA256']=sha(native/'weights/best.pt') if stage=='final50' else None
        active_sources=[];expected_sources=None;epoch_batches=0
        class GuardedNative11(DetectionTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                assert weights is not None and weights.model[-1].nc==10
                target=super().get_model(cfg,weights,verbose);src=weights.float().state_dict();dst=target.state_dict()
                assert set(src)==set(dst) and all(torch.equal(dst[k],v) for k,v in src.items())
                assert not target.end2end and target.model[-1].reg_max==16
                target.model[-1].max_det=1000
                return target
            def _build_train_pipeline(self):
                assert getattr(self,'_oom_retries',0)==0,'Do not silently halve batch after OOM'
                return super()._build_train_pipeline()
            def _handle_nan_recovery(self,epoch):
                assert_finite(self.loss);assert_finite(self.model.state_dict());assert_finite(self.ema.ema.state_dict())
                assert all(math.isfinite(float(v)) for v in self.metrics.values())
                return False
            def preprocess_batch(self,batch):
                if stage in ('short3','final50'):
                    assert batch['img'].dtype==torch.uint8
                    names=[Path(v).name for v in batch['im_file']];assert len(names)==len(batch['img'])
                    active_sources.extend(names);report['batches']+=1
                    with inputs.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=self.epoch+1,
                        batch=report['batches'],sources=names,CPU_raw_input_shape=list(batch['img'].shape)))+'\n')
                prepared=super().preprocess_batch(batch)
                if stage in ('short3','final50'):
                    nonlocal epoch_batches
                    epoch_batches+=1;self.model.criterion.observation_enabled=epoch_batches<=2
                    if epoch_batches==1:
                        import hashlib
                        digest=hashlib.sha256()
                        for key in ('img','cls','bboxes','batch_idx'):
                            value=prepared[key].detach().cpu().contiguous();digest.update(key.encode());digest.update(value.numpy().tobytes())
                        with (base/'prepared_first_batch_SHA.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=self.epoch+1,SHA256=digest.hexdigest(),sources=names))+'\n')
                return prepared
            def optimizer_step(self):
                row=observe_native_optimizer_step(self,super().optimizer_step)
                report['native_optimizer_attempts']+=1
                report['effective_optimizer_updates']+=int(row['effective_optimizer_step'])
                report['AMP_overflow_skips']+=int(row['amp_overflow_skip'])
                with observations.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=self.epoch+1,**row))+'\n')
                assert report['AMP_overflow_skips']<=32,'Stop for review; no precision/scaler recipe change'
            def save_model(self):
                assert_native_serialization_preconditions(self)
                return super().save_model()
            def final_eval(self):
                if stage in ('short3','final50'):
                    retained_best=native/'weights/selected_best_before_final_strip.pt';assert not retained_best.exists()
                    shutil.copyfile(self.best,retained_best)
                    ck=torch.load(retained_best,map_location='cpu',weights_only=False)
                    assert ck['epoch']+1==report['native_best_epoch'] and ck['ema'] is not None
                    report.update(selected_best_before_strip=str(retained_best),selected_best_before_strip_SHA256=sha(retained_best))
                    del ck
                report['native_final_eval_entered']=True;save()
                result=super().final_eval()
                report['native_final_eval_returned']=True;save();return result
        # Observe the actual unchanged AMP inner comparison. Reject skipped True.
        from ultralytics.engine import trainer as trainer_module
        original_amp=trainer_module.check_amp
        if settings['amp'] and stage!='setup_CPU':
            from yolo26_native_amp_observer_v1 import require_observed_native_amp_check
            def observed_amp(model):
                evidence={}
                try:return require_observed_native_amp_check(original_amp,model,evidence)
                finally:report['native_AMP_check_observation']=evidence;save()
            trainer_module.check_amp=observed_amp
        tr=GuardedNative11(overrides=settings)
        def setup_check(t):
            nonlocal expected_sources
            assert len(t.train_loader.dataset)==6471 and len(t.test_loader.dataset)==548
            assert t.args.epochs==t.epochs==50 and t.batch_size==settings['batch'] and t.args.imgsz==1280
            assert t.amp==settings['amp'] and not t.args.close_mosaic
            assert t.model.model[-1].reg_max==16 and not t.model.end2end
            t.model.criterion=t.model.init_criterion();assert type(t.model.criterion).__name__=='ObservedNativeLoss'
            assert t.model.stride.tolist()==protocol['expected_strides']
            actual_assigner=t.model.criterion.assigner.native
            assert type(actual_assigner).__name__==('SelectiveRankAssigner' if protocol['variant']=='C2_P2_selective_rank' else 'TaskAlignedAssigner')
            if protocol['variant']=='C2_P2_selective_rank':assert actual_assigner.rank_strength==.25 and actual_assigner.input_tiny_side==16.
            assert tensor_state_sha(t.model)==(expected_restored_state if stage=='final50' else initial_state)
            expected_sources={Path(v).name for v in t.train_loader.dataset.im_files};assert len(expected_sources)==6471
            if stage=='final50':
                assert t.start_epoch==3 and t.scheduler.last_epoch==2 and t.ema.updates==history['native_optimizer_attempts']
                assert len(t.optimizer.state)==len(expected_optimizer['state']) and t.scaler.state_dict()==expected_scaler
                # Native restore converts each optimizer tensor to the device;
                # verify its preserved payload with CPU values, including steps.
                state=t.optimizer.state_dict()
                assert state['param_groups']==expected_optimizer['param_groups']
                for k,original in expected_optimizer['state'].items():
                    for key,value in original.items():
                        restored=state['state'][k][key]
                        assert torch.equal(restored.cpu(),value.cpu()) if isinstance(value,torch.Tensor) else restored==value
            else:assert t.start_epoch==0 and not t.resume and not t.optimizer.state and t.ema.updates==0
            report.update(status='running',actual_settings=vars(t.args),actual_train_images=6471,actual_val_images=548,
                actual_amp=bool(t.amp),native_batches_per_epoch=len(t.train_loader),native_head='YOLO11 one2many reg_max16',
                resource_at_setup=resources(),framework_version=ultralytics.__version__,torch_version=torch.__version__)
            save()
        tr.add_callback('on_pretrain_routine_end',setup_check)
        def epoch_begin(t):
            nonlocal epoch_batches
            epoch_batches=0;active_sources.clear();step_deadline(release,started);resources()
        def after_batch(t):
            assert_finite(t.loss);assert_finite(t.loss_items)
            if t.model.criterion.observations:
                with (base/'bounded_assignment_diagnostics.jsonl').open('a') as f:
                    for item in t.model.criterion.observations:f.write(json.dumps(dict(epoch=t.epoch+1,epoch_batch=epoch_batches,**item),allow_nan=False)+'\n')
                t.model.criterion.observations.clear()
            if report['batches']%100==0:step_deadline(release,started);save()
        def val_guard(v):
            if v.training:assert_finite(v.loss)
        def saved_epoch(t):
            assert 1<=t.epoch+1<=50 and len(active_sources)==6471 and set(active_sources)==expected_sources
            assert all(math.isfinite(float(v)) for v in t.metrics.values())
            assert t.ema.updates==report['native_optimizer_attempts']
            best=sha(t.best)
            if best!=report['best_written_checkpoint_SHA256']:
                report['native_best_epoch']=t.epoch+1;report['best_written_checkpoint_SHA256']=best
            report['epochs'].append(dict(epoch=t.epoch+1,native_saved_unix=time.time(),
                source_count=len(active_sources),unique_sources=len(set(active_sources)),framework_metrics=t.metrics,
                native_optimizer_attempts=report['native_optimizer_attempts'],effective_optimizer_updates=report['effective_optimizer_updates'],
                AMP_overflow_skips=report['AMP_overflow_skips'],EMA_updates=t.ema.updates,
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            report['completed_training_epochs']=t.epoch+1
            if t.epoch+1==(3 if stage=='short3' else 50):
                retained=native/f'weights/retained_native_epoch{t.epoch+1}_before_strip.pt';assert not retained.exists()
                shutil.copyfile(t.last,retained)
                report.update(retained_checkpoint=str(retained),retained_checkpoint_SHA256=sha(retained))
                if stage=='short3':t.stop=True
            save()
        if stage=='setup_CPU':
            tr._setup_train()
            assert tr.device.type=='cpu' and not torch.cuda.is_available() and report['native_optimizer_attempts']==0
            report.update(status='completed_CPU_setup',passed=True,completed_unix=time.time(),no_backward_or_optimizer_step=True)
        elif stage=='capacity_GPU':
            tr._setup_train();tr.epoch=0
            raw=torch.load(ROOT/protocol['dense_fixture'],map_location='cpu',weights_only=False)
            batch=settings['batch'];valid=(raw['cls'].view(-1)>=0)&(raw['cls'].view(-1)<10)&(raw['batch_idx']<batch)
            raw['img']=raw['img'][:batch]
            for k in ('cls','bboxes','batch_idx'):raw[k]=raw[k][valid]
            assert raw['img'].shape==(batch,3,1280,1280)
            report['capacity_stress_target_counts']=[int((raw['batch_idx']==i).sum()) for i in range(batch)]
            report['capacity_fixture_limit']='Previously frozen native mosaic-augmented stress input; not research recipe or efficacy. Real whole-data stability requires short3.'
            for group in tr.optimizer.param_groups:group['lr']=.00001
            report['capacity_engineering_lr']=.00001
            tr.scaler=torch.cuda.amp.GradScaler(enabled=tr.amp,init_scale=128.)
            dense=tr.preprocess_batch(raw);torch.cuda.reset_peak_memory_stats()
            for attempt in range(8):
                step_deadline(release,started);tr.model.train();tr.optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=tr.amp):
                    tr.loss,tr.loss_items=tr.model(dense)
                assert_finite(tr.loss);assert_finite(tr.loss_items)
                tr.scaler.scale(tr.loss.sum()).backward();tr.optimizer_step()
                if report['effective_optimizer_updates']==2:break
            assert report['effective_optimizer_updates']==2
            del dense;tr.optimizer.zero_grad(set_to_none=True)
            metrics,fitness=tr.validate()
            assert all(math.isfinite(float(v)) for v in metrics.values()) and math.isfinite(float(fitness))
            report.update(status='completed_capacity_GPU',passed=True,completed_unix=time.time(),native_full548_validation_finite=True,
                parameters_and_optimizer_EMA_resident_during_validation=True,
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                formal_training_released=False)
        else:
            capacity=load(ROOT/protocol['paths']['capacity_GPU']/'run_manifest.json')
            assert capacity['status']=='completed_capacity_GPU' and capacity['passed'] and capacity['effective_optimizer_updates']==2
            assert capacity['protocol_SHA256']==sha(protocol_path)
            tr.add_callback('on_train_epoch_start',epoch_begin);tr.add_callback('on_train_batch_end',after_batch)
            tr.add_callback('on_val_end',val_guard);tr.add_callback('on_model_save',saved_epoch)
            tr.train()
            cap=3 if stage=='short3' else 50
            assert report['completed_training_epochs']==cap and [r['epoch'] for r in report['epochs']]==list(range(1,cap+1))
            assert report['batches']==cap*report['native_batches_per_epoch']
            assert report['native_final_eval_entered'] and report['native_final_eval_returned']
            best=tr.best;last=tr.last
            for ckpath in (best,last):
                ck=torch.load(ckpath,map_location='cpu',weights_only=False)
                assert ck['epoch']==-1 and ck['optimizer'] is None and ck['ema'] is None
                loaded,_=load_checkpoint(str(ckpath),device='cpu',fuse=False);assert_finite(loaded.state_dict());del ck,loaded
            report.update(status='completed_bounded_h50_short3' if stage=='short3' else 'completed_native11_h50_full50',
                passed=True,completed_unix=time.time(),best_checkpoint=str(best),best_checkpoint_SHA256=sha(best),
                last_checkpoint_SHA256=sha(last),results_csv_SHA256=sha(tr.csv),input_sources_SHA256=sha(inputs),
                optimizer_observations_SHA256=sha(observations),native_final_eval_executed=True)
            if stage=='short3':report['planning_after_actual_short3']=planning_gate(report,time.time())
        save()
        print(json.dumps(dict(status=report['status'],report=str(report_path),report_SHA256=sha(report_path))),flush=True)
    except Exception:
        report.update(status='failed_stop_no_retry',passed=False,error=traceback.format_exc(),failed_unix=time.time());save();raise
    finally:
        if 'trainer_module' in locals():trainer_module.check_amp=original_amp

def entry_main(stage,entry):
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    run(stage,entry,a.protocol,a.release)
