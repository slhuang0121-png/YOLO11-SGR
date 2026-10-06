"""Separate completed50 proof, source probe/full export and frozen CPU metrics.

No target domains or modern efficacy claims. Pending/failed training never
imports tensors or touches learned checkpoints. Each stage needs own release.
"""
from pathlib import Path
import argparse,ast,csv,hashlib,json,os,sys,time,traceback
from types import SimpleNamespace
from native11_h50_guard_v1 import ROOT,admit,atomic,load,sha,verify_pins,step_deadline

EXPORTER_SHA='bd47863f990926cb51c858a618660091998e7d41c4274b47967f094c79401b02'
METRIC_ENTRY_SHA='52376d33d1866645371f83b0158784ded722ad632ebfcd37343802140f439abe'
FOCI=('all','raw_lt16','raw_lt32','isolated_lt32','occlusion_0','occlusion_1','occlusion_2')
METRIC_PINS={
 'analysis/evaluation_metrics.py':'87b6b535c33c15b9a6fde358655775c0e392b2b3b7c45a15d6ef69d284c475d6',
 'engineering_code/roi_inference_20261002_v2/visdrone_person_adapter_v1.py':'d432ef8626215f74b7d5f5c57e8e05cca1f1f89a5355d94239fc8aaa64b6c46e',
 'engineering_code/roi_inference_20261002_v2/person_froc_v1.py':'7445ab1dc13de41ea49d9ce8fe2b9f99b8c9a2fa62e03442839fb2847b87bd3f',
 'engineering_code/roi_inference_20261002_v2/box_fusion_v1.py':'4950918d82144bc28d0b05c894adc7d0a5df788bef0002f8fb4e88e6cb33da46',
 'lab_modern/visdrone_toolkit_port.py':'2e0aae0c0d08860e6e291bf9ae1331bd43ea843db037a06d86e11a88b6de403e',
 'evidence/visdrone_runtime_parity.json':'ca80daff2f7382599c4898d8bd31ddd44b5785deb88ab07d9329057d631a513b'}

def actual_final(protocol):
    run=ROOT/protocol['paths']['final50'];p=run/'run_manifest.json'
    if not p.exists():return None,None,dict(status='pending_native11_h50_final50',no_checkpoint_read=True,no_tensor_imports=True)
    m=load(p)
    if m['status'] in ('starting','running'):return None,None,dict(status='pending_native11_h50_final50',no_checkpoint_read=True,no_tensor_imports=True)
    assert m['status']=='completed_native11_h50_full50' and m['passed'] and m.get('error') is None
    assert m['completed_training_epochs']==m['schedule_horizon']==50 and m['start_completed_epoch']==3 and m['additional_training_epochs']==47
    assert m['native_final_eval_entered'] and m['native_final_eval_returned'] and m['native_final_eval_executed']
    assert [r['epoch'] for r in m['epochs']]==list(range(1,51))
    assert all(r['source_count']==r['unique_sources']==6471 for r in m['epochs'])
    assert m['actual_train_images']==6471 and m['actual_val_images']==548
    return run,m,None

def completed_proof(protocol,protocol_path,run,m):
    p=ROOT/protocol['paths']['verify50_CPU']/'completion.json';d=load(p)
    assert d['status']=='completed' and d['passed'] and d['training_completed_epochs']==50
    assert d['protocol_SHA256']==sha(protocol_path) and d['actual_completed_manifest_SHA256']==sha(run/'run_manifest.json')
    assert d['checkpoint_SHA256']==m['best_checkpoint_SHA256'] and d['selected_checkpoint_actual_epoch']==m['native_best_epoch']
    assert d['source_SHA256']==sha(__file__)
    return p,d

def verify50(protocol,protocol_path,run,m,out,release,started):
    # Metadata and all source/exposure/update records before checkpoint load.
    assert m['protocol_SHA256']==sha(protocol_path)
    for filename,key in [('results.csv','results_csv_SHA256')]:assert sha(run/'native'/filename)==m[key]
    for filename,key in [('input_sources.jsonl','input_sources_SHA256'),('optimizer_observations.jsonl','optimizer_observations_SHA256')]:assert sha(run/filename)==m[key]
    rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader((run/'native/results.csv').open())]
    assert [int(float(r['epoch'])) for r in rows]==list(range(1,51))
    expected={p.name for p in (ROOT/'datasets/VisDrone/VisDrone2019-DET-train/images').glob('*.jpg')};assert len(expected)==6471
    counts=[[] for _ in range(50)];batches=0
    with (run/'input_sources.jsonl').open() as f:
        for line in f:
            row=json.loads(line);epoch=row['epoch'];assert 1<=epoch<=50
            batches+=1;assert row['batch']==batches
            assert row['CPU_raw_input_shape'][1:]==[3,1280,1280] and len(row['sources'])==row['CPU_raw_input_shape'][0]
            counts[epoch-1].extend(row['sources'])
    assert batches==m['batches']==50*m['native_batches_per_epoch']
    assert all(len(names)==6471 and set(names)==expected for names in counts)
    attempts=effective=skips=0
    with (run/'optimizer_observations.jsonl').open() as f:
        for line in f:
            row=json.loads(line);attempts+=1;effective+=int(row['effective_optimizer_step']);skips+=int(row['amp_overflow_skip'])
            assert row['native_ema_updates_after']==attempts and row['native_ema_updates_before']==attempts-1
    assert (attempts,effective,skips)==(m['native_optimizer_attempts'],m['effective_optimizer_updates'],m['AMP_overflow_skips'])
    for name,key in [('best.pt','best_checkpoint_SHA256'),('last.pt','last_checkpoint_SHA256')]:assert sha(run/'native/weights'/name)==m[key]
    assert sha(m['retained_checkpoint'])==m['retained_checkpoint_SHA256']
    assert sha(m['selected_best_before_strip'])==m['selected_best_before_strip_SHA256']
    import torch
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    from native11_h50_guard_v1 import tensor_state_sha
    retained=torch.load(m['retained_checkpoint'],map_location='cpu',weights_only=False)
    assert retained['epoch']==49 and retained['updates']==attempts and retained['optimizer'] is not None
    assert retained['train_args']['epochs']==50
    rawbest=torch.load(m['selected_best_before_strip'],map_location='cpu',weights_only=False)
    stripped=torch.load(run/'native/weights/best.pt',map_location='cpu',weights_only=False)
    assert rawbest['epoch']+1==m['native_best_epoch'] and stripped['epoch']==-1 and stripped['optimizer'] is None
    assert tensor_state_sha(rawbest['ema'].float())==tensor_state_sha(stripped['model'].float())
    del retained,rawbest,stripped
    step_deadline(release,started)
    result=dict(status='completed',passed=True,source_SHA256=sha(__file__),completed_unix=time.time(),
        actual_completed_manifest_SHA256=sha(run/'run_manifest.json'),protocol_SHA256=sha(protocol_path),
        checkpoint_SHA256=m['best_checkpoint_SHA256'],selected_checkpoint_actual_epoch=m['native_best_epoch'],
        training_completed_epochs=50,native_schedule_horizon=50,actual_input_batches=batches,
        each_source_once_per_epoch=True,native_optimizer_attempts=attempts,effective_optimizer_updates=effective,
        AMP_overflow_skips=skips,initial_VisDrone_training_epochs=0,native_final_eval_executed=True,
        native_best_before_and_after_strip_same_model_state=True,GPU_used=False,strict_architecture_causality=False)
    atomic(out/'completion.json',result)

def export(protocol,protocol_path,run,m,out,stage,release,started):
    proof_path,proof=completed_proof(protocol,protocol_path,run,m)
    base=ROOT/'accuracy_exports/yolo26m_user_budget50_h200_truncated_CPU_square1280_20261002_v1'
    references=[json.loads(s) for s in (base/'input_prediction_provenance.jsonl').read_text().splitlines()]
    assert len(references)==548
    if stage=='export548_CPU':
        probe=load(ROOT/protocol['paths']['export_probe2_CPU']/'export_manifest.json')
        assert probe['status']=='completed' and probe['images']==probe['exported_images']==2
        assert probe['checkpoint_SHA256']==m['best_checkpoint_SHA256'] and probe['completion_SHA256']==sha(proof_path)
    else:references=references[:2]
    assert sha(ROOT/'analysis/export_yolo_predictions.py')==EXPORTER_SHA
    sys.path.insert(0,str(ROOT/'analysis'));sys.path.insert(0,str(ROOT/'engineering_code'))
    import cv2,numpy as np,torch,export_yolo_predictions as numerical
    torch.set_num_threads(2);cv2.setNumThreads(1);assert not torch.cuda.is_available()
    best=run/'native/weights/best.pt';assert sha(best)==m['best_checkpoint_SHA256']
    model,head,info=numerical.load_model(best,'one2many','cpu',1000,True)
    assert head.reg_max==16 and head.nc==10 and not head.end2end
    d=dict(status='running',purpose='engineering' if len(references)==2 else 'research',started_unix=started,
        source_SHA256=sha(__file__),numerical_exporter_SHA256=EXPORTER_SHA,protocol_SHA256=sha(protocol_path),
        checkpoint_SHA256=sha(best),completion_SHA256=sha(proof_path),training_completed_epochs=50,
        original_schedule_horizon=50,images=len(references),exported_images=0,precision='fp32',device='cpu',GPU_used=False,
        imgsz=1280,conf=.001,iou=.7,max_det=1000,multi_label=True,latency_claim=False,**info)
    report=out/'export_manifest.json';atomic(report,d);corpus=hashlib.sha256();provenance=out/'input_prediction_provenance.jsonl'
    with provenance.open('x') as f,torch.inference_mode():
        for ref in references:
            step_deadline(release,started)
            image_path=ROOT/'datasets/VisDrone/VisDrone2019-DET-val/images'/ref['image']
            annotation=image_path.parent.parent/'annotations'/(image_path.stem+'.txt')
            assert sha(image_path)==ref['source_image_sha256'] and sha(annotation)==ref['raw_annotation_sha256']
            image=cv2.imread(str(image_path));assert image is not None and list(image.shape[:2])==ref['original_shape']
            tensor,params=numerical.prepare_image(image,1280)
            rows=numerical.prediction_rows(model,head,tensor,params,.001,.7,1000,True)
            assert rows.shape[1]==8 and len(rows)<=1000 and np.isfinite(rows).all()
            assert sha(image_path)==ref['source_image_sha256'] and sha(annotation)==ref['raw_annotation_sha256']
            prediction=out/(image_path.stem+'.txt');np.savetxt(prediction,rows,fmt='%.12g',delimiter=',')
            corpus.update(prediction.name.encode());corpus.update(b'\0');corpus.update(prediction.read_bytes());corpus.update(b'\0')
            record=dict(image=ref['image'],source_image_sha256=ref['source_image_sha256'],raw_annotation_sha256=ref['raw_annotation_sha256'],
                original_shape=ref['original_shape'],prediction_name=prediction.name,prediction_sha256=sha(prediction),native_prediction_rows=len(rows),
                person_rows_category1or2=int(((rows[:,5]==1)|(rows[:,5]==2)).sum()))
            f.write(json.dumps(record)+'\n');f.flush();d['exported_images']+=1
            if d['exported_images']%25==0:atomic(report,d)
    assert sha(best)==m['best_checkpoint_SHA256']
    d.update(status='completed',completed_unix=time.time(),prediction_rows_sha256=corpus.hexdigest(),
             input_prediction_provenance_sha256=sha(provenance),exact_reference_images_and_raw_GT_preserved=True)
    atomic(report,d)

def metrics(protocol,protocol_path,run,m,out,release,started):
    completion_path,completion=completed_proof(protocol,protocol_path,run,m)
    export_dir=ROOT/protocol['paths']['export548_CPU'];export_path=export_dir/'export_manifest.json';export=load(export_path)
    assert export['status']=='completed' and export['images']==export['exported_images']==548
    assert export['source_SHA256']==sha(__file__) and export['completion_SHA256']==sha(completion_path)
    assert export['checkpoint_SHA256']==m['best_checkpoint_SHA256']
    verify_pins(METRIC_PINS);parent=ROOT/'analysis/evaluate_paired_h50_completed50_person_cpu_v1.py';assert sha(parent)==METRIC_ENTRY_SHA
    protected={str(ROOT/k):v for k,v in METRIC_PINS.items()}
    protected.update({str(parent):METRIC_ENTRY_SHA,str(export_path):sha(export_path),str(completion_path):sha(completion_path),
                      str(run/'run_manifest.json'):sha(run/'run_manifest.json'),str(Path(__file__)):sha(__file__)})
    files=sorted(export_dir.glob('*.txt'));assert len(files)==548;corpus=hashlib.sha256()
    for p in files:corpus.update(p.name.encode());corpus.update(b'\0');corpus.update(p.read_bytes());corpus.update(b'\0');protected[str(p)]=sha(p)
    assert corpus.hexdigest()==export['prediction_rows_sha256']
    provenance_file=export_dir/'input_prediction_provenance.jsonl';assert sha(provenance_file)==export['input_prediction_provenance_sha256']
    protected[str(provenance_file)]=sha(provenance_file)
    source_rows=[json.loads(s) for s in provenance_file.read_text().splitlines()]
    assert len(source_rows)==len({r['image'] for r in source_rows})==548
    assert {r['prediction_name'] for r in source_rows}=={p.name for p in files}
    dataset=ROOT/'datasets/VisDrone/VisDrone2019-DET-val'
    for row in source_rows:
        assert Path(row['image']).name==row['image'] and row['image'].endswith('.jpg')
        assert row['prediction_name']==Path(row['image']).stem+'.txt'
        assert 0<=row['person_rows_category1or2']<=row['native_prediction_rows']<=1000
        assert sha(export_dir/row['prediction_name'])==row['prediction_sha256']
        assert sha(dataset/'images'/row['image'])==row['source_image_sha256']
        assert sha(dataset/'annotations'/(Path(row['image']).stem+'.txt'))==row['raw_annotation_sha256']
        protected[str(dataset/'images'/row['image'])]=row['source_image_sha256']
        protected[str(dataset/'annotations'/(Path(row['image']).stem+'.txt'))]=row['raw_annotation_sha256']
    sys.path.insert(0,str(ROOT/'engineering_code/roi_inference_20261002_v2'));sys.path.insert(0,str(ROOT/'analysis'))
    import numpy as np
    from evaluation_metrics import load_visdrone,verified_author_metrics,coco_metrics
    from visdrone_person_adapter_v1 import prepare_frame
    from person_froc_v1 import curve
    assert 'torch' not in sys.modules
    images,gt,categories,dt,prepared,provenance=load_visdrone(dataset,export_dir)
    assert len(images)==len(prepared)==548 and provenance['prediction_rows_sha256']==corpus.hexdigest()
    args=SimpleNamespace(strategy='native11_P2_matched_'+protocol['variant']+'_native_best',output=out/'person_FROC7_two_AP.json')
    result=dict(status='running',started_unix=started,strategy=args.strategy,source_SHA256=sha(__file__),froc={},
        checkpoint_SHA256=m['best_checkpoint_SHA256'],export_manifest_SHA256=sha(export_path),protocol_SHA256=sha(protocol_path),
        training_completed_epochs=50,original_schedule_horizon=50,selected_checkpoint_actual_epoch=m['native_best_epoch'],
        actual_completed_proof_SHA256=sha(completion_path),provenance=provenance,GPU_used=False,
        matched_input_or_strict_architecture_causality_claim=False,algorithm_efficacy_established=False,
        warning='Development checkpoint selection and thresholds; seven strata overlap and each stratum threshold is descriptive. Cross-domain policy must use all-person source threshold. Native AMP/batch/loss/single-view differences disclosed.')
    def checkpoint():atomic(args.output,result)
    checkpoint()
    # Execute the exact frozen numerical try block, with a new metadata scope.
    # It contains all seven strata, untouched NPZ arrays, author500 and COCO1000.
    tree=ast.parse(parent.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    block=next(n for n in main.body if isinstance(n,ast.Try))
    assert isinstance(block.body[0],ast.FunctionDef) and block.body[0].name=='already_filtered'
    assert isinstance(block.body[1],ast.For) and isinstance(block.body[1].iter,ast.Name) and block.body[1].iter.id=='FOCI'
    scope=dict(ROOT=ROOT,FOCI=FOCI,np=np,images=images,gt=gt,categories=categories,dt=dt,prepared=prepared,
        result=result,output_dir=out,args=args,checkpoint=checkpoint,prepare_frame=prepare_frame,curve=curve,
        verified_author_metrics=verified_author_metrics,coco_metrics=coco_metrics,sha=sha,protected=protected,time=time,json=json)
    exec(compile(ast.Module(body=block.body,type_ignores=[]),'<unchanged-seven-FROC-two-AP>', 'exec'),scope)
    assert result['status']=='completed' and set(result['froc'])==set(FOCI)
    step_deadline(release,started)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['verify50_CPU','export_probe2_CPU','export548_CPU','metrics_CPU'],required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--release',type=Path,required=True);a=p.parse_args()
    protocol,release=admit(a.protocol,a.release,a.stage,__file__)
    run,m,pending=actual_final(protocol)
    if pending is not None:print(json.dumps(pending));return
    assert m['protocol_SHA256']==sha(a.protocol)
    out=ROOT/protocol['paths'][a.stage];assert not out.exists();out.mkdir(parents=True)
    started=time.time();os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        YOLO_CONFIG_DIR=str(out/'config'),NO_ALBUMENTATIONS_UPDATE='1')
    try:
        if a.stage=='verify50_CPU':verify50(protocol,a.protocol,run,m,out,release,started)
        elif a.stage=='metrics_CPU':metrics(protocol,a.protocol,run,m,out,release,started)
        else:export(protocol,a.protocol,run,m,out,a.stage,release,started)
    except Exception:
        atomic(out/'actual_failure.json',dict(status='failed_stop_no_retry',source_SHA256=sha(__file__),stage=a.stage,
            error=traceback.format_exc(),failed_unix=time.time(),existing_partial_files_preserved=True));raise
if __name__=='__main__':main()
