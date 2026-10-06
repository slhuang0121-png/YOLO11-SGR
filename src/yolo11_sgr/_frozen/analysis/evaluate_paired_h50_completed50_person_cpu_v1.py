"""True paired h50/50 CPU development metrics, after sealed complete548 export.

Admission/provenance changes only; the seven person-FROC computation block
and published-toolkit/COCO calls retain the previously verified metric core.
Pending runs/exports return before arrays, predictions, checkpoints or output.
No training, GPU execution, rescue-test claim or candidate benefit is implied.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time

ROOT = Path('/root/autodl-tmp/rescuetiny_revision')
VERIFIER_SHA = '12917149f6b22c82c458dc4b2512b15036b2c83a8903fe702e0fb6fc4f105126'
EXPORT_SOURCE_SHA = '5cdcc3f96ee7c80e6c5cb0102f8b06749e37b25b507f00cfe3ac3266707a74c0'
PEER_SOURCE_SHA = 'dd0f7d2cb138b32c213aad3637d01ebd33f389891ffc67a857c16b9da833805e'
ORIGINAL_METRIC_ENTRY_SHA = '8ab5d1c6596398eb74dce3e2cf2070c6e922307100ebe8bfa90f1986f138bf2f'
FOCI = ('all','raw_lt16','raw_lt32','isolated_lt32','occlusion_0','occlusion_1','occlusion_2')
METRIC_PINS = {
    'analysis/evaluation_metrics.py':'87b6b535c33c15b9a6fde358655775c0e392b2b3b7c45a15d6ef69d284c475d6',
    'engineering_code/roi_inference_20261002_v2/visdrone_person_adapter_v1.py':'d432ef8626215f74b7d5f5c57e8e05cca1f1f89a5355d94239fc8aaa64b6c46e',
    'engineering_code/roi_inference_20261002_v2/person_froc_v1.py':'7445ab1dc13de41ea49d9ce8fe2b9f99b8c9a2fa62e03442839fb2847b87bd3f',
    'engineering_code/roi_inference_20261002_v2/box_fusion_v1.py':'4950918d82144bc28d0b05c894adc7d0a5df788bef0002f8fb4e88e6cb33da46',
    'lab_modern/visdrone_toolkit_port.py':'2e0aae0c0d08860e6e291bf9ae1331bd43ea843db037a06d86e11a88b6de403e',
    'evidence/visdrone_runtime_parity.json':'ca80daff2f7382599c4898d8bd31ddd44b5785deb88ab07d9329057d631a513b',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def validate_export_contract(manifest,phase):
    """Pure metadata contract; does not stand in for actual SHA/provenance."""
    assert manifest['status']=='completed' and manifest['purpose']=='research' and manifest['phase']==phase
    assert manifest['images']==manifest['exported_images']==548
    assert manifest['source_sha256']==EXPORT_SOURCE_SHA
    assert manifest['training_completed_epochs']==manifest['original_training_schedule_horizon']==50
    assert manifest['original_training_status']=='completed_h50_native_research_50'
    assert manifest['original_native_final_eval_executed'] is True
    assert manifest['same_pause_epoch']==3 and manifest['additional_completed_epochs']==47
    assert manifest['device']=='cpu' and manifest['GPU_used'] is False and manifest['precision']=='fp32'
    assert manifest['imgsz']==1280 and manifest['input_tensor_shape']==[1,3,1280,1280]
    assert manifest['conf']==.001 and manifest['iou']==.7 and manifest['max_det']==1000
    assert manifest['multi_label'] is True and manifest['mode']=='one2many' and manifest['fused'] is True
    assert manifest['one2many_NMS_IoU']==.7
    assert manifest['exact_reference_pixels_and_original_annotations_preserved'] is True
    assert manifest['paired_strength']==(0. if phase=='ordinary_paired' else 1.)
    assert manifest['training_started'] is False and manifest['latency_claim'] is False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['ordinary_paired','grouped_paired'],required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-admission',action='store_true')
    modes.add_argument('--evaluate',action='store_true')
    args = parser.parse_args()
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    helper = ROOT/'engineering_code/verify_paired_native_h50_research50_completion_v1.py'
    assert sha(helper)==VERIFIER_SHA
    spec = importlib.util.spec_from_file_location('paired50_person_completion_admission',helper)
    verification = importlib.util.module_from_spec(spec);spec.loader.exec_module(verification)
    assert sha(ROOT/'evidence/current_user_experiment_budget.json')==verification.BUDGET_SHA
    phase = args.phase
    run = ROOT/'research_runs'/f'paired26m_{phase}_s42_h50_resume3_to50_i1280_FP32_20261002_v2'
    run_path = run/'run_manifest.json'
    run_manifest = json.loads(run_path.read_text()) if run_path.exists() else None
    gate = verification.completion_gate(run_manifest)
    export = ROOT/'accuracy_exports'/f'paired26m_{phase}_h50_cumulative50_CPU_square1280_20261003_v1'
    export_path = export/'export_manifest.json'
    completion_path = ROOT/'evidence'/f'paired_h50_{phase}_research50_completion_verified_20261003_v1.json'
    pending = None
    if gate!='eligible_for_actual_CPU_verification':
        pending = gate
    elif not completion_path.exists():
        pending = 'pending_sealed_CPU_completion'
    elif not export_path.exists():
        pending = 'pending_complete548_export'
    else:
        export_manifest = json.loads(export_path.read_text())
        if export_manifest['status'] in ('starting','running'):
            pending = 'pending_complete548_export'
        else:
            assert export_manifest['status']=='completed',export_manifest.get('error',export_manifest['status'])
    if pending is not None:
        print(json.dumps(dict(status=pending,phase=phase,no_array_or_tensor_imports=True,
            no_active_checkpoint_read=True,no_prediction_or_GT_read=True,
            no_metric_report_written=True,no_evaluation_started=True,no_training_launch=True)))
        return
    completion = json.loads(completion_path.read_text())
    assert completion['passed'] and completion['status']=='completed' and completion['phase']==phase
    assert completion['verifier_source_SHA256']==VERIFIER_SHA
    assert completion['actual_completed_manifest_SHA256']==sha(run_path)
    assert completion['actual_full_training_epochs']==completion['native_schedule_horizon']==50
    assert completion['native_final_eval_flag_and_log_verified'] and not completion['automatic_next_phase_allowed']
    assert completion['actual_input_batches']==161800 and completion['each_source_exactly_once_per_epoch']
    assert completion['initial_VisDrone_training_epochs']==0 and completion['initial_tensor_state_SHA256']==verification.INITIAL_TENSOR_SHA
    assert completion['training_source_SHA256']==run_manifest['source_SHA256']==verification.SOURCE_SHA
    validate_export_contract(export_manifest,phase)
    assert export_manifest['actual_completed_proof_SHA256']==sha(completion_path)
    assert export_manifest['training_run_manifest_SHA256']==sha(run_path)
    assert export_manifest['selected_checkpoint_actual_epoch']==completion['native_best_epoch']
    assert export_manifest['selected_checkpoint_epoch_candidates']==completion['native_best_epoch_candidates_from_preserved_metrics']
    assert Path(export_manifest['checkpoint']).resolve()==(run/'weights/best.pt').resolve()
    assert export_manifest['checkpoint_sha256']==run_manifest['best_checkpoint_SHA256']
    protected = dict(completion['protected_files_SHA256'])
    for p,h in export_manifest['protected_sources_SHA256'].items():
        assert p not in protected or protected[p]==h,('Shared source mismatch',p)
        protected[p]=h
    protected.update({str(completion_path):sha(completion_path),str(export_path):sha(export_path),
        str(Path(__file__)):sha(__file__),str(helper):VERIFIER_SHA})
    for relative,h in METRIC_PINS.items():
        protected[str(ROOT/relative)]=h
    original_metric_entry = ROOT/'analysis/evaluate_yolo26m_budget50_person_cpu_v1.py'
    assert sha(original_metric_entry)==ORIGINAL_METRIC_ENTRY_SHA
    protected[str(original_metric_entry)]=ORIGINAL_METRIC_ENTRY_SHA
    assert sha(ROOT/'analysis/export_paired_yolo26m_h50_completed50_cpu_v1.py')==EXPORT_SOURCE_SHA
    predictions = export
    files = sorted(predictions.glob('*.txt'));assert len(files)==548
    corpus = hashlib.sha256()
    for file in files:
        corpus.update(file.name.encode());corpus.update(b'\0');corpus.update(file.read_bytes());corpus.update(b'\0')
        protected[str(file)]=sha(file)
    assert corpus.hexdigest()==export_manifest['prediction_rows_sha256']
    provenance_file = export/'input_prediction_provenance.jsonl'
    assert sha(provenance_file)==export_manifest['input_prediction_provenance_sha256']
    protected[str(provenance_file)]=sha(provenance_file)
    input_records = [json.loads(line) for line in provenance_file.read_text().splitlines()]
    assert len(input_records)==548 and len({row['image'] for row in input_records})==548
    assert {row['prediction_name'] for row in input_records}=={p.name for p in files}
    dataset = ROOT/'datasets/VisDrone/VisDrone2019-DET-val'
    for row in input_records:
        assert Path(row['image']).name==row['image'] and row['image'].endswith('.jpg')
        assert row['prediction_name']==Path(row['image']).stem+'.txt'
        assert 0<=row['native_prediction_rows']<=1000
        assert 0<=row['person_rows_category1or2']<=row['native_prediction_rows']
        protected[str(dataset/'images'/row['image'])]=row['source_image_sha256']
        protected[str(dataset/'annotations'/(Path(row['image']).stem+'.txt'))]=row['raw_annotation_sha256']
        assert sha(export/row['prediction_name'])==row['prediction_sha256']
    peer_path = ROOT/'evidence/paired_h50_all161800_inputs_and_updates_matched_20261003_v1.json'
    peer_SHA = None
    if peer_path.exists():
        peer = json.loads(peer_path.read_text())
        assert peer['passed'] and peer['comparison_source_SHA256']==PEER_SOURCE_SHA
        assert peer['actual_completed_epochs_each']==50 and peer['all161800_actual_input_GT_tensor_SHA_sources_order_and_draws_exact']
        assert peer['phases'][phase]['actual_completed_manifest_SHA256']==sha(run_path)
        peer_SHA=sha(peer_path);protected[str(peer_path)]=peer_SHA
    for p,h in protected.items():assert sha(p)==h,p
    if args.check_admission:
        print(json.dumps(dict(status='ready_for_actual_CPU_development_evaluation',phase=phase,
            source_SHA256=sha(__file__),export_manifest_SHA256=sha(export_path),completion_proof_SHA256=sha(completion_path),
            no_array_or_tensor_imports=True,no_evaluation_started=True,no_metric_report_written=True,
            actual_complete548_corpus_and_provenance_verified=True,paired_peer_proof_SHA256=peer_SHA)))
        return
    args.strategy = f'paired26m_{phase}_h50_cumulative50_native_best'
    output_dir = ROOT/'research_evaluations'/f'paired26m_{phase}_h50_cumulative50_person_20261003_v1'
    args.output = output_dir/(args.strategy+'.json')
    assert not output_dir.exists()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
    sys.path.insert(0,str(ROOT/'engineering_code/roi_inference_20261002_v2'))
    sys.path.insert(0,str(ROOT/'analysis'))
    import inspect, numpy as np
    from evaluation_metrics import load_visdrone,verified_author_metrics,coco_metrics
    from visdrone_person_adapter_v1 import prepare_frame
    from person_froc_v1 import curve
    assert 'torch' not in sys.modules
    assert Path(inspect.getfile(prepare_frame)).resolve()==(ROOT/'engineering_code/roi_inference_20261002_v2/visdrone_person_adapter_v1.py').resolve()
    assert Path(inspect.getfile(curve)).resolve()==(ROOT/'engineering_code/roi_inference_20261002_v2/person_froc_v1.py').resolve()
    images,gt,categories,dt,prepared,provenance=load_visdrone(dataset,predictions)
    assert len(images)==len(prepared)==548 and provenance['prediction_rows_sha256']==corpus.hexdigest()
    result=dict(status='running',strategy=args.strategy,phase=phase,purpose='development evaluation; not independent final test',
        frozen_metric_code_SHA256=METRIC_PINS,training_completed_epochs=50,original_schedule_horizon=50,
        selected_checkpoint_actual_epoch=completion['native_best_epoch'],
        selected_checkpoint_epoch_candidates=completion['native_best_epoch_candidates_from_preserved_metrics'],
        comparison_label=export_manifest['comparison_label'],paired_strength=export_manifest['paired_strength'],
        matched_fresh_h50_primary_comparison=peer_SHA is not None,paired_peer_completion_proof_SHA256=peer_SHA,
        actual_completed_proof_SHA256=sha(completion_path),training_run_manifest_SHA256=sha(run_path),
        initial_VisDrone_training_epochs=0,initial_tensor_state_SHA256=verification.INITIAL_TENSOR_SHA,
        same_pause_epoch=3,additional_completed_training_epochs=47,native_CSV_time_resets_at_resume=True,
        reused_frozen_metric_core_source_sha256='0fd6c96e45278a26aeae1657e8a33b1bc4b96fb1503911c27f5ab667e4a0bffb',
        prior_capped50_metric_entry_SHA256=ORIGINAL_METRIC_ENTRY_SHA,
        checkpoint_sha256=export_manifest['checkpoint_sha256'],export_manifest_sha256=sha(export_path),
        export_prediction_corpus_sha256=export_manifest['prediction_rows_sha256'],provenance=provenance,
        source_sha256=sha(Path(__file__)),started_unix=time.time(),froc={},protected_sources_SHA256=protected,
        array_imports_after_actual_completed_export_admission=True,GPU_used=False,training_or_resume_called=False,
        algorithm_efficacy_established=False,
        warning='Confidence thresholds below are selected on this development set only; final or cross-domain operating points must reuse them without selecting on new GT. Strata overlap and each stratum selects its own development threshold; no causal occlusion claim. A single condition report cannot establish candidate benefit.',
        scope='True h50/cumulative50 selected native checkpoint; original548 VisDrone development annotations/ignore protocol, all10 globalcap1000 beforeperson1/2. Seven raw-size/isolation/occlusion person FROC strata plus published500 continuous-VOC and COCO101 max1000 kept separate. No ROI or target-GT threshold tuning, and no independent SAR/disaster-test or deployment-latency claim. Native rounded resume and time reset disclosed.')
    output_dir.mkdir(parents=True)
    def checkpoint():atomic(args.output,result)
    checkpoint()
    try:
        # Unchanged numerical block from the actually completed old reference
        # evaluator. Its AST is independently checked, including all seven
        # strata, ignore flag conversion and lossless curve array serialization.
        def already_filtered(gt,dt,height,width):return gt,dt
        for focus in FOCI:
            frames=[]
            for image,(official_gt,filtered_dt) in zip(images,prepared):
                raw_flag_gt=official_gt.copy();raw_flag_gt[:,4]=1-raw_flag_gt[:,4]
                frames.append(prepare_frame(raw_flag_gt,filtered_dt,image['height'],image['width'],already_filtered,focus))
            metric=curve(frames)
            points=metric.pop('curve')
            curve_path=output_dir/(args.strategy+'_person_'+focus+'_steps.npz');assert not curve_path.exists()
            np.savez_compressed(curve_path,threshold=np.array([np.inf if r['threshold'] is None else r['threshold'] for r in points]),
                tp=np.array([r['tp'] for r in points],dtype=np.int64),fp=np.array([r['fp'] for r in points],dtype=np.int64),
                fp_empty=np.array([r['fp_on_no_evaluable_person_images'] for r in points],dtype=np.int64),
                images=len(images),target_count=metric['target_count'])
            metric.update(curve_step_count=len(points),curve_file=curve_path.name,curve_sha256=sha(curve_path),
                          ignore_filter_applied_once_by_verified_load_visdrone=True)
            result['froc'][focus]=metric;checkpoint()
            print(json.dumps(dict(strategy=args.strategy,focus=focus,targets=metric['target_count'],
                recall_at_fpi_05=metric['reference_fpi_steps']['0.5']['recall'])),flush=True)
            del frames,points,metric
        parity=ROOT/'evidence/visdrone_runtime_parity.json'
        result['published_visdrone_toolkit']=verified_author_metrics(prepared,parity);checkpoint()
        result['coco_cap1000_auxiliary']=coco_metrics(images,gt,categories,dt,1000)
        for p,h in protected.items():assert sha(p)==h,p
        result.update(status='completed',completed_unix=time.time(),all_protected_sources_unchanged=True)
        checkpoint();print(json.dumps(dict(strategy=args.strategy,status='completed')),flush=True)
    except Exception as exc:
        result.update(status='failed',failed_unix=time.time(),error=repr(exc))
        checkpoint()
        raise


if __name__=='__main__':main()
