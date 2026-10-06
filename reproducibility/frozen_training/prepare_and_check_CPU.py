"""Original C0 initializer, known-P2 C1, native real-fixture CPU train/restore.

This is a bounded engineering run, never a formal 50-epoch efficacy result.
The output directory is exclusive; failed runs are preserved for review.
"""
import argparse, copy, hashlib, json, os, random, sys, time, traceback, zipfile
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def atomic(p, d):
    p = Path(p); temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'); temp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    assert out.is_relative_to(root) and not out.exists()
    out.mkdir(parents=True)
    for name in ('prepare_and_check_CPU.py', 'model_and_trainer.py'):
        (out/name).write_bytes((Path(__file__).parent/name).read_bytes())
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        YOLO_CONFIG_DIR=str(out/'config'), NO_ALBUMENTATIONS_UPDATE='1')
    report = dict(status='running', started_unix=time.time(), pid=os.getpid(),
        GPU_used=False, server_access=False, formal_training_released=False,
        new_accuracy_result=False, source_SHA256=sha(__file__), phases={})
    rp = out/'actual_CPU_preparation_report.json'
    atomic(rp, report)
    try:
        import numpy as np, torch, ultralytics, yaml
        from types import SimpleNamespace
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel, load_checkpoint
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from model_and_trainer import append_p2, as_observed_native, tensor_sha, MatchedNativeTrainer, ObservedNativeLoss
        torch.set_num_threads(2)
        assert not torch.cuda.is_available() and ultralytics.__version__ == '8.4.170'
        previous = root/'现代实验代码/native11_person_semantic_matched_h50_20261004_v1/reviewed_protocol.json'
        p = json.loads(previous.read_text(encoding='utf-8'))
        lib = Path(ultralytics.__file__).parent
        for rel, digest in p['native_source_SHA256'].items(): assert sha(lib/rel) == digest, rel
        canonical = root/'真实实验结果/作者数据与切片审阅完整证据20261002_v1/datasets/VisDrone/deim_native_effective_labels_20261002_v2'
        train_file, val_file = canonical/'instances_train_zero_based.json', canonical/'instances_val_zero_based.json'
        assert sha(train_file) == 'dd0ba1370298c826ffe822f539ab194463fbc3620d488d55c0c0e0b12d5f40db'
        train = json.loads(train_file.read_text(encoding='utf-8')); val = json.loads(val_file.read_text(encoding='utf-8'))
        assert len(train['images']) == 6471 and len(train['annotations']) == 343200
        assert len(val['images']) == 548 and len(val['annotations']) == 38759
        names = {c['id']: c['name'] for c in train['categories']}
        official = root/'server/downloads/yolo11m.pt'
        assert sha(official) == 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        source = YOLO(str(official)).model.float().cpu()
        torch.manual_seed(42)
        recipe = copy.deepcopy(p['recipe'])
        cfg = get_cfg(overrides=dict(recipe, device='cpu', workers=0))
        base = DetectionModel(copy.deepcopy(source.yaml), nc=10, verbose=False)
        factory = SimpleNamespace(args=cfg, data=dict(names=names, nc=10))
        base = DetectionTrainer.set_model_names_for_load(factory, base); base.load(source, verbose=False)
        assert tensor_sha(base) == p['matched_baseline']['zero_update_tensor_SHA256']
        base.args = cfg; base.nc = 10; base.names = names; base.model[-1].max_det = 1000; base.criterion = None
        c0 = as_observed_native(base)
        c1, mapping, extra = append_p2(base)
        baseline_hash = tensor_sha(c0)
        assert baseline_hash == '0e153526905c7895cf5c1eb1f34a6b3070750db033d010cb96e1e9b8ffd9fc8f'
        checkpoints = {}
        for name, model in [('C0_native11', c0), ('C1_append_P2', c1)]:
            checkpoint = out/(name+'_zero_updates_FP32.pt')
            torch.save(dict(model=model, epoch=-1, ema=None, optimizer=None, updates=0,
                train_args=vars(cfg), version=ultralytics.__version__), checkpoint)
            restored, ck = load_checkpoint(str(checkpoint), device='cpu', fuse=False)
            assert ck['epoch'] == -1 and ck['optimizer'] is None and tensor_sha(restored.float()) == tensor_sha(model)
            checkpoints[name] = dict(path=str(checkpoint), SHA256=sha(checkpoint), tensor_SHA256=tensor_sha(model))
        # Observation must leave raw losses and all parameter gradients unchanged.
        c1.args = cfg; c1.eval(); c1.model[-1].training = True
        x = torch.linspace(0, 1, 3*64*64).reshape(1, 3, 64, 64)
        batch = dict(img=x, batch_idx=torch.tensor([0., 0.]), cls=torch.tensor([[0.], [3.]]),
            bboxes=torch.tensor([[.5,.5,.125,.125], [.8,.8,.125,.125]]))
        c1.criterion = c1.init_criterion()
        losses = []; gradients = []
        for enabled in (False, True):
            c1.zero_grad(set_to_none=True); c1.criterion.observation_enabled = enabled
            loss, items = c1(batch)
            vs = list(loss.values()) if isinstance(loss, dict) else [loss]
            total = sum(v.sum() for v in vs); assert torch.isfinite(total)
            total.backward(); losses.append(total.detach().clone())
            gradients.append({k:v.grad.detach().clone() for k,v in c1.named_parameters() if v.grad is not None})
        assert torch.equal(losses[0], losses[1])
        assert set(gradients[0]) == set(gradients[1]) and all(torch.equal(v, gradients[1][k]) for k,v in gradients[0].items())
        assert c1.criterion.observations and c1.criterion.assigner.last is None
        atomic(out/'synthetic_assignment_observation.json', c1.criterion.observations)
        c1.criterion = None; c1.zero_grad(set_to_none=True)
        report['phases']['initialization_and_observer'] = dict(passed=True, baseline_exact_tensor_SHA256=baseline_hash,
            common_tensors=len(mapping), only_new_P2_tensors=extra, checkpoints=checkpoints,
            observation_loss_and_every_gradient_bit_exact=True,
            semantic_person_row_transfer_used=False, baseline_parameters=sum(v.numel() for v in c0.parameters()),
            candidate_parameters=sum(v.numel() for v in c1.parameters()))
        atomic(rp, report)
        # Eight training and two validation original images, exclusive fixtures.
        by_image = {}
        for ann in train['annotations']: by_image.setdefault(ann['image_id'], []).append(ann)
        order = train['images']
        chosen = order[:6]
        for image in sorted(order, key=lambda im:len(by_image.get(im['id'], [])), reverse=True):
            if image not in chosen:
                chosen.append(image); break
        for image in order:
            if image not in chosen and not by_image.get(image['id']):
                chosen.append(image); break
        if len(chosen) < 8:
            chosen.append(next(im for im in order if im not in chosen))
        fixture = out/'fixture'; receipts = []
        for split, document, images, archive_name in [('train',train,chosen,'VisDrone2019-DET-train.zip'),
                ('val',val,val['images'][:2],'VisDrone2019-DET-val.zip')]:
            anns = {}
            for ann in document['annotations']: anns.setdefault(ann['image_id'], []).append(ann)
            images_dir = fixture/split/'images'; labels_dir = fixture/split/'labels'
            images_dir.mkdir(parents=True); labels_dir.mkdir(parents=True)
            zpath = root/'server/downloads'/archive_name
            with zipfile.ZipFile(zpath) as archive:
                for image in images:
                    member = 'VisDrone2019-DET-'+split+'/images/'+image['file_name']
                    raw = archive.read(member); destination=images_dir/image['file_name']; destination.write_bytes(raw)
                    lines=[]
                    for a in anns.get(image['id'], []):
                        xx,yy,ww,hh=a['bbox']; width,height=image['width'],image['height']
                        lines.append(' '.join([str(a['category_id'])]+[format(v,'.17g') for v in ((xx+ww/2)/width,(yy+hh/2)/height,ww/width,hh/height)]))
                    label = labels_dir/(Path(image['file_name']).stem+'.txt')
                    label.write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
                    receipts.append(dict(split=split, canonical_image_id=image['id'], archive_member=member,
                        input_image_SHA256=sha(destination), fixture_label_SHA256=sha(label), canonical_annotations=len(lines)))
        data = dict(path=str(fixture), train='train/images', val='val/images', names=names, nc=10, channels=3)
        data_file=out/'engineering_fixture_data.yaml'; data_file.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
        atomic(out/'actual_fixture_sources.json', dict(scope='Canonical converted labels, original ZIP image bytes; 8+2 engineering only',
            canonical_train_SHA256=sha(train_file),canonical_val_SHA256=sha(val_file),receipts=receipts))
        # Native train(), validation, optimizer, EMA, checkpoint, and setup restore.
        runs={}
        for name in ('C0_native11','C1_append_P2'):
            inputs=[]; observations=[]; updates=[]; retained=[]
            class CPUTrainer(MatchedNativeTrainer):
                def preprocess_batch(self, b):
                    prepared=super().preprocess_batch(b)
                    h=hashlib.sha256()
                    for k in ('img','batch_idx','cls','bboxes'):
                        t=prepared[k].detach().cpu().contiguous();h.update(k.encode());h.update(t.numpy().tobytes())
                    inputs.append(dict(epoch=self.epoch+1,image_names=[Path(f).name for f in prepared['im_file']],prepared_input_SHA256=h.hexdigest()))
                    if self.model.criterion is None: self.model.criterion=self.model.init_criterion()
                    self.model.criterion.observation_enabled=True
                    return prepared
                def optimizer_step(self):
                    super().optimizer_step()
                    updates.append(self.ema.updates)
                def save_model(self):
                    super().save_model()
                    snapshot=self.save_dir/('unstripped_epoch'+str(self.epoch+1)+'.pt')
                    snapshot.write_bytes(self.last.read_bytes()); retained.append(str(snapshot))
            settings=dict(recipe,model=checkpoints[name]['path'],data=str(data_file),device='cpu',workers=0,
                amp=False,imgsz=128,plots=False,save=True,val=True,project=str(out),name=name,exist_ok=False)
            tr=CPUTrainer(overrides=settings)
            def finish(t):
                observations.extend(t.model.criterion.observations);t.model.criterion.observations.clear()
                if t.epoch==1:t.stop=True
            tr.add_callback('on_train_epoch_end',finish)
            tr.train()
            assert tr.epoch==1 and tr.args.epochs==50 and len(inputs)==4 and updates
            for point in inputs: assert len(point['image_names'])==4
            saved_path=Path(retained[-1]);saved=torch.load(saved_path,map_location='cpu',weights_only=False)
            assert saved['epoch']==1 and saved['optimizer'] is not None and saved['ema'] is not None and saved['updates']>0
            resume_dir=out/(name+'_restore_setup')
            restored_tr=CPUTrainer(overrides=dict(settings,model=str(saved_path),resume=str(saved_path),project=str(resume_dir),name='native',exist_ok=False))
            restored_tr._setup_train()
            assert restored_tr.start_epoch==2 and restored_tr.args.epochs==50
            assert restored_tr.scheduler.last_epoch==1 and restored_tr.ema.updates==saved['updates']
            assert len(restored_tr.optimizer.state)==len(saved['optimizer']['state'])
            atomic(out/(name+'_assignment_diagnostics.json'),observations)
            runs[name]=dict(scope='2 native CPU engineering epochs; 8 train/2 val; 128px; FP32; workers0; no efficacy claim',
                training_batch_records=inputs, native_optimizer_steps=len(updates), EMA_updates=tr.ema.updates,
                unstripped_checkpoint=str(saved_path), checkpoint_SHA256=sha(saved_path),
                optimizer_and_EMA_retained=True, resume_setup_passed=True, resume_start_epoch=restored_tr.start_epoch,
                formal_recipe_changes=['device CPU','AMP false','workers0','128px','8 train/2 val subset','stop after 2 engineering epochs'],
                diagnostic_object_rows=sum(len(x['rows']) for x in observations))
            report['phases']['native_CPU_train_restore_partial']=copy.deepcopy(runs)
            atomic(rp,report)
            del tr,restored_tr,saved
        assert runs['C0_native11']['training_batch_records']==runs['C1_append_P2']['training_batch_records']
        assert runs['C0_native11']['native_optimizer_steps']==runs['C1_append_P2']['native_optimizer_steps']
        report.update(status='completed',passed=True,completed_unix=time.time(),phases=dict(report['phases'],native_CPU_train_restore=runs),
            actual_real_fixture_inputs_matched=True,native_source_SHA256={rel:sha(lib/rel) for rel in list(p['native_source_SHA256'])+['utils/tal.py','data/augment.py']},
            server_GPU_release=False, pending=['Server data/code identity','CUDA1280 capacity and actual AMP','Full-data own short3','Actual CUDA optimizer/EMA/scaler restore','Reviewed cumulative50 continuation','548 and rescue efficacy and cost'])
        atomic(rp,report);print(json.dumps(dict(status=report['status'],passed=True,report=str(rp)),ensure_ascii=False))
    except Exception:
        report.update(status='failed',passed=False,error=traceback.format_exc(),failed_unix=time.time());atomic(rp,report);raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
