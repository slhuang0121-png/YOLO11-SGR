"""Audit completed retained rescue evidence and compare two frozen source policies.

No model inference, training, threshold selection, SSH or input mutation.
Readiness is stdlib only. Audit loads one event frame at a time and reproduces
every retained NPZ score group, including ignored detections and empty frames.
"""
from pathlib import Path
import argparse
import csv
import ctypes
import hashlib
import json
import math
import sys
import time

R = Path('D:/论文/返修')
C0 = R/'研究短试验/新匹配C0完整50救援1857本机CPU评测20261005_v1'
C2 = R/'研究短试验/新匹配C2完整50救援1857本机CPU评测20261005_v1'
OUT = R/'真实实验结果/C0_C2完整50救援1857两固定源政策实际比较20261005_v1'
FOCI = ('all','raw_lt16','raw_lt32','isolated_lt32')
PINS = {
    C0/'metrics/rescue_FROC_source_fixed.json': 'b02c211a2a278cd9b18e8bd22fb70df585571e1a8a755a322c329b2393ef6e45',
    C2/'protocol_before_target_export.json': '09ec8be8a11a9f4d6041237f624f5e244b81839690f48928e4d9fa597e76babf',
    R/'scripts/C2_matched50_rescue_local_CPU_20261005_v1.py': 'ce3b4e88723c61d161aa460d57c5196c3d22b3cae4a4dbf640b81a004134ba30',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def available():
    class State(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    value=State();value.length=ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.avail_phys

START=time.time()
def guard():
    assert time.time()-START<900, 'Bounded evidence audit exceeded 900 seconds'
    assert available()>=750*1024*1024, 'Leave 750 MiB local memory available'
    assert 'torch' not in sys.modules

def readiness():
    manifest=C2/'full1857/export_manifest.json'
    value=read(manifest) if manifest.exists() else {}
    metrics=C2/'metrics/rescue_FROC_source_fixed.json'
    result=read(metrics) if metrics.exists() else {}
    ready=(value.get('status')=='completed' and value.get('exported_images')==1857
           and result.get('status')=='completed' and result.get('passed') is True)
    return dict(status='ready_for_evidence_audit' if ready else 'pending_preserve_current_export_or_metrics',
                ready=ready,export_status=value.get('status'),exported_images=value.get('exported_images'),
                metrics_status=result.get('status'),model_or_tensor_loaded=False,output_directory_created=False)

def provenance(directory, protocol, manifest, before, verify_images):
    for domain in ('bari','sds'):
        path=directory/'full1857'/domain/'input_prediction_provenance.jsonl'
        assert sha(path)==manifest['domains'][domain]['provenance_SHA256']
        before[str(path)]=sha(path)
        expected={q['image']:q for q in protocol['image_records'][domain]}
        seen=set();corpus=hashlib.sha256()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                guard();q=json.loads(line);assert q['image'] not in seen;seen.add(q['image'])
                source=expected[q['image']]
                assert q['source_image_SHA256']==source['source_image_SHA256']
                assert Path(q['path'])==Path(source['path'])
                if verify_images:
                    assert sha(q['path'])==q['source_image_SHA256']
                prediction=directory/'full1857'/domain/q['prediction_name']
                assert sha(prediction)==q['prediction_SHA256']
                corpus.update(prediction.name.encode());corpus.update(b'\0')
                corpus.update(prediction.read_bytes());corpus.update(b'\0')
        assert seen==set(expected)
        assert corpus.hexdigest()==manifest['domains'][domain]['prediction_corpus_SHA256']

def baseline_metadata(protocol, report, before):
    result={}
    for domain in ('bari','sds'):
        path=C0/'metrics'/(domain+'_per_frame_froc_events.jsonl')
        assert sha(path)==report['per_frame_event_SHA256'][domain]
        before[str(path)]=sha(path)
        expected={(q['image_id'],q['image']) for q in protocol['image_records'][domain]}
        seen={focus:set() for focus in FOCI}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                guard();q=json.loads(line);f=q['frame'];focus=q['focus']
                identity=(q['image_id'],q['image']);assert identity in expected and identity not in seen[focus]
                seen[focus].add(identity)
                result[domain,focus,*identity]={k:v for k,v in f.items() if k!='events'}
        assert all(group==expected for group in seen.values())
    return result

def audit_events(np, domain, focus, protocol, report, metadata, before):
    path=C2/'metrics'/(domain+'_per_frame_froc_events.jsonl')
    assert sha(path)==report['per_frame_event_SHA256'][domain]
    before[str(path)]=sha(path)
    scenarios=('all','mountains','beaches') if domain=='bari' else ('all',)
    expected={(q['image_id'],q['image']) for q in protocol['image_records'][domain]}
    curves={}
    for scenario in scenarios:
        v=report['summaries'][domain][scenario][focus]
        p=C2/'metrics'/v['curve_file'];assert sha(p)==v['curve_SHA256'];before[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:
            arrays={key:z[key].copy() for key in ('threshold','tp','fp','fp_empty')}
            n=int(z['images']);gt=int(z['target_count'])
        threshold=arrays['threshold']
        assert threshold[0]==np.inf and np.all(np.isfinite(threshold[1:]))
        assert np.all(np.diff(threshold)<0) and len(threshold)==v['curve_step_count']
        assert n==v['images'] and gt==v['target_count']
        for key in ('tp','fp','fp_empty'):
            assert arrays[key][0]==0 and np.all(np.diff(arrays[key])>=0)
        curves[scenario]=dict(value=v,arrays=arrays,neg=-threshold,
            increments={key:np.zeros(len(threshold),dtype=np.int64) for key in ('tp','fp','fp_empty','events')},
            images=0,gt=0,empty=0,points=[[0,0,0],[0,0,0]])
    seen=set();taus=[report['source_threshold'],report['secondary_source_policy']['threshold']]
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            q=json.loads(line)
            if q['focus']!=focus:
                continue
            guard();f=q['frame'];identity=(q['image_id'],q['image'])
            assert identity in expected and identity not in seen;seen.add(identity)
            assert {k:v for k,v in f.items() if k!='events'}==metadata[domain,focus,*identity]
            assert f['no_evaluable_person']==(f['evaluable_person_count']==0)
            assert 0<=f['target_count']<=f['evaluable_person_count']
            events=f['events'];scores=np.array([e['score'] for e in events],dtype=np.float64)
            tps=np.array([e['tp'] for e in events],dtype=np.int64)
            fps=np.array([e['fp'] for e in events],dtype=np.int64)
            assert np.all(np.isfinite(scores)) and np.all((scores>=0)&(scores<=1))
            assert np.all((tps==0)|(tps==1)) and np.all((fps==0)|(fps==1)) and np.all(tps+fps<=1)
            assert int(tps.sum())<=f['target_count']
            chosen=['all']+([f['scenario']] if domain=='bari' else [])
            for scenario in chosen:
                c=curves[scenario];c['images']+=1;c['gt']+=f['target_count'];c['empty']+=f['no_evaluable_person']
                indices=np.searchsorted(c['neg'],-scores)
                assert np.all(indices<len(c['neg']))
                assert np.array_equal(c['arrays']['threshold'][indices],scores)
                for key,values in [('tp',tps),('fp',fps),('fp_empty',fps*int(f['no_evaluable_person'])),('events',np.ones(len(events),dtype=np.int64))]:
                    np.add.at(c['increments'][key],indices,values)
                for i,tau in enumerate(taus):
                    selected=scores>=tau
                    c['points'][i][0]+=int(tps[selected].sum())
                    c['points'][i][1]+=int(fps[selected].sum())
                    c['points'][i][2]+=int(fps[selected].sum())*int(f['no_evaluable_person'])
    assert seen==expected
    rows=[]
    for scenario,c in curves.items():
        v=c['value'];assert c['images']==v['images'] and c['gt']==v['target_count'] and c['empty']==v['no_evaluable_person_images']
        assert np.all(c['increments']['events'][1:]>0)
        for key in ('tp','fp','fp_empty'):
            assert np.array_equal(np.cumsum(c['increments'][key]),c['arrays'][key]),(domain,scenario,focus,key)
        for i,budget in enumerate((0.5,1.0)):
            op=v['frozen_threshold_operating_point'] if i==0 else v['secondary_source_1FPI_operating_point']
            tp,fp,fp_empty=c['points'][i]
            computed=dict(threshold=taus[i],tp=tp,fp=fp,recall=tp/c['gt'] if c['gt'] else None,
                fpi=fp/c['images'],fp_on_no_evaluable_person_images=fp_empty,
                mean_fp_per_no_evaluable_person_image=fp_empty/c['empty'] if c['empty'] else None)
            assert all(op[k]==value for k,value in computed.items())
            ix=int(np.flatnonzero(c['arrays']['threshold']>=taus[i])[-1])
            assert [int(c['arrays'][k][ix]) for k in ('tp','fp','fp_empty')]==[tp,fp,fp_empty]
            rows.append(dict(domain=domain,scenario=scenario,stratum=focus,source_budget=budget,
                             images=c['images'],targets=c['gt'],C2_point=computed))
    return rows

def main(stage):
    ready=readiness()
    if stage=='readiness' or not ready['ready']:
        print(json.dumps(ready,ensure_ascii=True));return
    assert not OUT.exists(), 'Successful comparison already exists; do not repeat'
    guard()
    for p,digest in PINS.items():assert sha(p)==digest
    p0=read(C0/'protocol_before_target_export.json');p2=read(C2/'protocol_before_target_export.json')
    assert p0['image_records']==p2['image_records'] and p0['image_inventories']==p2['image_inventories']
    assert p2['both_source_policies_frozen_before_any_target_export'] and p2['target_GT_used_for_selection'] is False
    for key in ('conf','iou','max_det','multi_label','size','head','precision','fuse','CPU_threads','GPU_used'):
        assert p0[key]==p2[key],key
    r0=read(C0/'metrics/rescue_FROC_source_fixed.json');r2=read(C2/'metrics/rescue_FROC_source_fixed.json')
    assert r0['status']==r2['status']=='completed' and r0['passed'] and r2['passed']
    assert r2['primary_source_policy_not_replaced'] and not r2['target_labels_used_for_model_or_threshold_selection']
    assert r2['source_threshold']==0.492431968451 and r2['secondary_source_policy']['threshold']==0.410106301308
    assert r2['source_metric_SHA256']=='075bb842743a977729f95a3bc8e01e0c7a0f4108d3b6c834bb7e6a40831d1fe5'
    assert r2['checkpoint_SHA256']=='5a8717d200fd9a73178913130682684788f42d04aa81e5ce061e44fa4a253977'
    before={str(p):sha(p) for p in PINS}
    for directory,protocol,report in ((C0,p0,r0),(C2,p2,r2)):
        guard();ep=directory/'full1857/export_manifest.json';manifest=read(ep)
        assert sha(ep)==report['export_manifest_SHA256']
        assert manifest['status']=='completed' and manifest['exported_images']==manifest['images']==1857
        assert manifest['all_protected_files_unchanged'] and report['all_protected_files_unchanged']
        assert manifest['checkpoint_SHA256']==protocol['checkpoint_SHA256']==report['checkpoint_SHA256']
        assert sha(directory/'protocol_before_target_export.json')==report['protocol_SHA256']
        before[str(ep)]=sha(ep);before[str(directory/'protocol_before_target_export.json')]=sha(directory/'protocol_before_target_export.json')
        before[str(directory/'metrics/rescue_FROC_source_fixed.json')]=sha(directory/'metrics/rescue_FROC_source_fixed.json')
        for p,digest in protocol['protected_files'].items():assert sha(p)==digest
        provenance(directory,protocol,manifest,before,verify_images=directory==C2)
    prior=C0/'secondary_source1_from_retained_original_curves/actual_C0_secondary_source1_transfer_points.json'
    secondary=read(prior);before[str(prior)]=sha(prior)
    assert secondary['status']=='completed_original_curves_primary_reproduced_secondary_source1'
    assert secondary['C2_transfer_protocol_SHA256']==sha(C2/'protocol_before_target_export.json')
    assert secondary['source_threshold']==0.428935348988
    assert all(sha(p)==digest for p,digest in secondary['input_SHA256'].items())
    baseline={(q['domain'],q['scenario'],q['stratum']):q for q in secondary['rows']}
    metadata=baseline_metadata(p0,r0,before)
    import numpy as np
    rows=[]
    for domain in ('bari','sds'):
        for focus in FOCI:
            rows.extend(audit_events(np,domain,focus,p2,r2,metadata,before))
    for row in rows:
        base=baseline[row['domain'],row['scenario'],row['stratum']]
        assert (row['images'],row['targets'])==(base['images'],base['targets'])
        q=base['original_primary_source_policy_point'] if row['source_budget']==0.5 else base['secondary_source1_point']
        row['C0_point']=q
        row['C2_minus_C0_recall_percentage_points']=None if q['recall'] is None else 100*(row['C2_point']['recall']-q['recall'])
        row['C2_minus_C0_false_positives']=row['C2_point']['fp']-q['fp']
        row['C2_minus_C0_target_FPI']=row['C2_point']['fpi']-q['fpi']
    assert len(rows)==32
    assert all(sha(p)==digest for p,digest in before.items())
    guard();OUT.mkdir()
    result=dict(status='completed_exact_target_curves_events_and_two_source_policies_verified',
        created_unix=time.time(),elapsed_seconds=time.time()-START,script_SHA256=sha(__file__),
        input_SHA256=before,full1857_input_source_GT_and_numerical_settings_match=True,
        C2_all_original_curve_arrays_reproduced_from_original_per_frame_events=True,
        both_source_policies_frozen_before_C2_target_inference=True,target_threshold_tuning=False,
        primary_policy_not_replaced=True,target_FPI_not_forced_equal_to_source_budget=True,
        target_FPI_not_matched_across_models=True,empty_strata_preserved_as_null=True,
        no_new_model_inference=True,no_significance_or_blind_independent_disaster_claim=True,rows=rows)
    with (OUT/'actual_C0_C2_two_source_policy_transfer_comparison.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    flat=[]
    for row in rows:
        q={k:v for k,v in row.items() if k not in ('C0_point','C2_point')}
        for arm in ('C0','C2'):
            q.update({arm+'_'+k:v for k,v in row[arm+'_point'].items()})
        flat.append(q)
    fields=list(dict.fromkeys(k for row in flat for k in row))
    with (OUT/'all_signed_transfer_strata_two_source_policies.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(status=result['status'],all_and_tiny=[q for q in rows if q['scenario']=='all' and q['stratum'] in ('all','raw_lt16')]),ensure_ascii=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['readiness','audit_compare'])
    main(parser.parse_args().stage)
