"""Independent C2 engineering, reusing completed C1 fixture receipts."""
import argparse,copy,hashlib,json,os,sys,time,traceback
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fixture,out=a.fixture_run.resolve(),a.output.resolve()
    assert not out.exists();out.mkdir(parents=True)
    for n in ('model_and_trainer.py','selective_rank.py','check_selective_rank_CPU.py'):
        (out/n).write_bytes((Path(__file__).parent/n).read_bytes())
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',YOLO_CONFIG_DIR=str(out/'config'),NO_ALBUMENTATIONS_UPDATE='1')
    report=dict(status='running',started_unix=time.time(),GPU_used=False,formal_training_released=False,accuracy_gain_claim=False,
        source_SHA256=sha(__file__),fixture_report_SHA256=sha(fixture/'actual_CPU_preparation_report.json'))
    rp=out/'actual_selective_rank_CPU_report.json'
    def save():rp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        import torch
        from types import SimpleNamespace
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.tal import TaskAlignedAssigner
        from model_and_trainer import MatchedNativeTrainer,tensor_sha
        from selective_rank import SelectiveRankAssigner,RankedDetectionModel
        torch.set_num_threads(2);assert not torch.cuda.is_available()
        previous=json.loads((fixture/'actual_CPU_preparation_report.json').read_text(encoding='utf-8'))
        assert previous['status']=='completed' and previous['passed']
        init=previous['phases']['initialization_and_observer']['checkpoints']['C1_append_P2']
        assert sha(init['path'])==init['SHA256']
        model,ck=load_checkpoint(init['path'],device='cpu',fuse=False);model=model.float()
        assert tensor_sha(model)==init['tensor_SHA256'] and ck['epoch']==-1 and ck['optimizer'] is None
        if isinstance(model.args,dict):model.args=SimpleNamespace(**model.args)
        model.__class__=RankedDetectionModel;model.rank_strength=.25;model.criterion=None
        # Known proposals, same candidate geometry, mixed classes, and empty GT.
        torch.manual_seed(29)
        anchors=torch.stack(torch.meshgrid(torch.arange(2.,64.,4.),torch.arange(2.,64.,4.),indexing='ij'),-1).reshape(-1,2)
        centers=anchors+torch.randn_like(anchors)*4
        wh=4+torch.rand_like(anchors)*20
        pred_boxes=torch.cat((centers-wh/2,centers+wh/2),-1).unsqueeze(0)
        scores=torch.rand((1,len(anchors),10))*.8+.1
        boxes=torch.tensor([[[20.,20.,28.,28.],[34.,34.,46.,46.],[2.,2.,46.,46.]]])
        labels=torch.tensor([[[0.],[3.],[1.]]]);valid=torch.ones((1,3,1))
        params=dict(topk=10,num_classes=10,alpha=.5,beta=6.,stride=[8.,16.,32.,4.],topk2=None)
        native=TaskAlignedAssigner(**params);zero=SelectiveRankAssigner(**params,strength=0);active=SelectiveRankAssigner(**params,strength=.25)
        args=(scores,pred_boxes,anchors,labels,boxes,valid)
        x=native(*args);y=zero(*args);z=active(*args)
        assert all(torch.equal(v,w) for v,w in zip(x,y))
        assert all(torch.isfinite(v).all() for v in z)
        # No tiny person: exact full native assignment output, including scores.
        other=labels.clone();other[0,0,0]=3
        nx=native(scores,pred_boxes,anchors,other,boxes,valid);ax=active(scores,pred_boxes,anchors,other,boxes,valid)
        assert all(torch.equal(v,w) for v,w in zip(nx,ax))
        # The returned quality metric and conflict overlap are native regardless
        # of changed ranking; verify the actual three tensors directly.
        active(*args);native(*args)
        p1,a1,o1=native.get_pos_mask(scores,pred_boxes,labels,boxes,anchors,valid)
        p2,a2,o2=active.get_pos_mask(scores,pred_boxes,labels,boxes,anchors,valid)
        assert torch.equal(a1,a2) and torch.equal(o1,o2)
        assert torch.equal(p1[:,1:],p2[:,1:])
        report.update(zero_strength_every_assignment_output_exact=True,no_tiny_person_every_assignment_output_exact=True,
            native_quality_and_conflict_metric_exact=True,noneligible_preconflict_masks_exact=True,
            eligible_preconflict_mask_changed_count=int((p1[:,:1]!=p2[:,:1]).sum()),
            synthetic_final_foreground_changed_count=int((x[3]!=z[3]).sum()),
            selection_definition='category0/1 and augmented INPUT area<256, not raw-original object identity',
            strength=.25,regression_CIoU_DFL_unchanged=True,shared_P2_initializer_tensor_SHA256=tensor_sha(model))
        checkpoint=out/'C2_selective_rank_zero_updates_FP32.pt'
        torch.save(dict(model=model,epoch=-1,ema=None,optimizer=None,updates=0,train_args=vars(model.args),version='8.4.170'),checkpoint)
        readback,_=load_checkpoint(str(checkpoint),device='cpu',fuse=False)
        if isinstance(readback.args,dict):readback.args=SimpleNamespace(**readback.args)
        assert readback.rank_strength==.25 and tensor_sha(readback.float())==tensor_sha(model)
        assert type(readback.init_criterion().assigner.native).__name__=='SelectiveRankAssigner'
        report.update(checkpoint=str(checkpoint),checkpoint_SHA256=sha(checkpoint),pickle_model_and_custom_criterion_readback_passed=True)
        save()
        inputs=[];observed=[];steps=[];retained=[]
        class EngineeringTrainer(MatchedNativeTrainer):
            def preprocess_batch(self,b):
                q=super().preprocess_batch(b);h=hashlib.sha256()
                for k in ('img','batch_idx','cls','bboxes'):
                    t=q[k].detach().cpu().contiguous();h.update(k.encode());h.update(t.numpy().tobytes())
                inputs.append(dict(epoch=self.epoch+1,image_names=[Path(f).name for f in q['im_file']],prepared_input_SHA256=h.hexdigest()))
                if self.model.criterion is None:self.model.criterion=self.model.init_criterion()
                self.model.criterion.observation_enabled=True
                return q
            def optimizer_step(self):
                super().optimizer_step();steps.append(self.ema.updates)
            def save_model(self):
                super().save_model();p=self.save_dir/('unstripped_epoch'+str(self.epoch+1)+'.pt');p.write_bytes(self.last.read_bytes());retained.append(str(p))
        settings=copy.deepcopy(vars(model.args));settings.update(model=str(checkpoint),data=str(fixture/'engineering_fixture_data.yaml'),
            device='cpu',amp=False,imgsz=128,workers=0,project=str(out),name='C2_engineering',plots=False,exist_ok=False)
        tr=EngineeringTrainer(overrides=settings)
        def stop_after_two(t):
            observed.extend(t.model.criterion.observations);t.model.criterion.observations.clear()
            if t.epoch==1:t.stop=True
        tr.add_callback('on_train_epoch_end',stop_after_two);tr.train()
        expected=previous['phases']['native_CPU_train_restore']['C1_append_P2']['training_batch_records']
        assert tr.epoch==1 and inputs==expected and steps
        raw=torch.load(retained[-1],map_location='cpu',weights_only=False)
        assert raw['epoch']==1 and raw['optimizer'] is not None and raw['ema'] is not None
        restore=EngineeringTrainer(overrides=dict(settings,model=retained[-1],resume=retained[-1],project=str(out/'restore_setup'),name='native',exist_ok=False))
        restore._setup_train();assert restore.start_epoch==2 and restore.model.rank_strength==.25
        assert restore.ema.updates==raw['updates'] and len(restore.optimizer.state)==len(raw['optimizer']['state'])
        (out/'assignment_diagnostics.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        report.update(status='completed',passed=True,completed_unix=time.time(),native_CPU_fixture_2_epochs_and_restore_passed=True,
            prepared_inputs_bit_exact_to_C1_fixture=True,native_optimizer_updates=len(steps),EMA_updates=tr.ema.updates,
            diagnostic_object_rows=sum(len(row['rows']) for row in observed),fixture_scope='8 train/2 val;128px;FP32;workers0;horizon50;stop2',
            pending=['Own CUDA capacity/AMP','Full-data bounded short3, not accuracy extrapolation','Own CUDA resume','Actual efficacy vs C0/C1 and rank-only ablation','False-positive and cross-domain cost evaluation'])
        save();print(json.dumps(dict(passed=True,report=str(rp)),ensure_ascii=False))
    except Exception:
        report.update(status='failed',passed=False,error=traceback.format_exc(),failed_unix=time.time());save();raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
