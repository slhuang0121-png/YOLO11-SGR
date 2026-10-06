"""Execute one separately reviewed stage; no queue, releases or GPU lock bypass."""
from pathlib import Path
import argparse,json,os,subprocess,time
from native11_h50_guard_v1 import ROOT,RESULT_END,load,sha,verify_pins

ENTRIES={
 'init_CPU':'native11_h50_init_CPU_v1.py','setup_CPU':'native11_h50_setup_CPU_v1.py',
 'capacity_GPU':'native11_h50_capacity_GPU_v1.py','short3':'native11_h50_short3_v1.py',
 'restore_CPU':'native11_h50_restore_probe_v1.py','restore_GPU':'native11_h50_restore_probe_v1.py',
 'final50':'native11_h50_resume3_to50_v1.py',
 'verify50_CPU':'native11_h50_verify_export_metrics_CPU_v1.py',
 'export_probe2_CPU':'native11_h50_verify_export_metrics_CPU_v1.py',
 'export548_CPU':'native11_h50_verify_export_metrics_CPU_v1.py','metrics_CPU':'native11_h50_verify_export_metrics_CPU_v1.py'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=ENTRIES,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);a=ap.parse_args()
    p,r=load(a.protocol),load(a.release)
    assert p['status']=='reviewed_protocol' and r['approved'] is True and r['stage']==a.stage
    assert p['kind']=='native11_P2_matched_h50_preparation_20261004_v1'
    assert r['protocol_SHA256']==sha(a.protocol) and not r['automatic_next_job_allowed']
    verify_pins(p['required_file_SHA256'])
    if r['evidence_SHA256']:verify_pins(r['evidence_SHA256'])
    for n,d in p['source_bundle_SHA256'].items():assert sha(Path(__file__).parent/n)==d,n
    entry=Path(__file__).parent/ENTRIES[a.stage];assert r['entry_source_SHA256']==sha(entry)
    now=time.time();timeout=r['max_wall_seconds']
    assert r['authorized_start_unix']<=now<r['authorized_end_unix']<=RESULT_END
    assert isinstance(timeout,int) and 0<timeout<r['authorized_end_unix']-now-120
    log=ROOT/'evidence'/('native11_P2_matched_'+p['variant']+'_'+a.stage+'_20261004_v1.log')
    receipt=log.with_suffix('.launch.json');assert not log.exists() and not receipt.exists()
    arguments=['--protocol',str(a.protocol),'--release',str(a.release)]
    if a.stage in ('restore_CPU','restore_GPU','verify50_CPU','export_probe2_CPU','export548_CPU','metrics_CPU'):
        arguments=['--stage',a.stage]+arguments
    # The child admission takes the original flock itself. Do not acquire a
    # second independent lock here, which would deadlock the protected child.
    command=['timeout','--signal=TERM','--kill-after=15s',str(timeout)+'s',
        str(ROOT/'env_modern/bin/python'),'-u',str(entry),*arguments]
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(entry.parent),str(ROOT/'engineering_code'),env.get('PYTHONPATH','')])
    env['CUDA_VISIBLE_DEVICES']='0' if a.stage in ('capacity_GPU','short3','restore_GPU','final50') else ''
    with log.open('xb') as f:
        child=subprocess.Popen(command,cwd=str(entry.parent),env=env,stdin=subprocess.DEVNULL,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,stage=a.stage,variant=p['variant'],started_unix=now,protocol_SHA256=sha(a.protocol),
        release_SHA256=sha(a.release),entry_SHA256=sha(entry),max_wall_seconds=timeout,
        GPU_lock='Same original gpu_queue.lock in child admission',automatic_next_job_allowed=False)
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
