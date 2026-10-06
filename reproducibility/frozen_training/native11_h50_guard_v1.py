"""Independent standard-library stage admission. Never auto-releases a job."""
from pathlib import Path
import datetime,hashlib,json,math,os,subprocess,sys,time

ROOT=Path('/root/autodl-tmp/rescuetiny_revision')
RESULT_END=datetime.datetime(2026,10,5,20,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
_LOCK=None
STAGES=('init_CPU','setup_CPU','capacity_GPU','short3','restore_CPU','restore_GPU','final50','verify50_CPU','export_probe2_CPU','export548_CPU','metrics_CPU')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def atomic(p,d):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)


def inside(relative):
    p=(ROOT/relative).resolve()
    assert not Path(relative).is_absolute() and p.is_relative_to(ROOT.resolve()),relative
    return p


def verify_pins(pins):
    assert isinstance(pins,dict) and pins
    for relative,digest in pins.items():
        assert len(digest)==64 and sha(inside(relative))==digest,relative


def resources():
    stat=os.statvfs(ROOT)
    assert stat.f_bavail*stat.f_frsize>=3*1024**3,'Preserve disk margin; do not delete originals'
    cgroup=Path('/sys/fs/cgroup');maximum=(cgroup/'memory.max').read_text().strip()
    current=int((cgroup/'memory.current').read_text())
    if maximum!='max':assert int(maximum)-current>=12*1024**3
    return dict(read_unix=time.time(),disk_available_bytes=stat.f_bavail*stat.f_frsize,
        RAM_current_bytes=current,RAM_limit=maximum,memory_events=(cgroup/'memory.events').read_text())


def tensor_state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def admit(protocol_path,release_path,stage,entry):
    global _LOCK
    assert 'torch' not in sys.modules and stage in STAGES
    p,r=load(protocol_path),load(release_path)
    assert p['kind']=='native11_P2_matched_h50_preparation_20261004_v1'
    assert p['variant'] in ('C0_native11','C1_append_P2','C2_P2_selective_rank')
    assert p['formal_maximum_epochs']==p['recipe']['epochs']==50
    assert p['result_deadline_unix']==RESULT_END and not p['automatic_next_job_allowed']
    assert r['approved'] is True and r['stage']==stage and r['rationale']
    assert r['protocol_SHA256']==sha(protocol_path) and r['entry_source_SHA256']==sha(entry)
    assert r['automatic_next_job_allowed'] is False
    now=time.time()
    assert r['authorized_start_unix']<=now<r['authorized_end_unix']<=RESULT_END
    assert 0<r['max_wall_seconds']<=r['authorized_end_unix']-now-60
    for name,digest in p['source_bundle_SHA256'].items():assert sha(Path(entry).parent/name)==digest,name
    for rel,digest in p['required_file_SHA256'].items():assert sha(inside(rel))==digest,rel
    for rel,digest in r['evidence_SHA256'].items():assert sha(inside(rel))==digest,rel
    # One independently reviewed phase, under the SAME original GPU lock.
    import fcntl
    _LOCK=(ROOT/'gpu_queue.lock').open('a+')
    fcntl.flock(_LOCK.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources()
    gpu=stage in ('capacity_GPU','short3','restore_GPU','final50')
    if gpu:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        os.environ['CUDA_VISIBLE_DEVICES']='0'
    else:os.environ['CUDA_VISIBLE_DEVICES']=''
    for rel in p['paths'].values():
        target=inside(rel);assert p['variant'] in rel and not target.is_symlink()
    if stage not in ('init_CPU','setup_CPU'):
        proof=load(inside(p['paths']['initialization'])/'initialization_proof.json')
        assert proof['passed'] and proof['protocol_SHA256']==sha(protocol_path)
        assert proof['tensor_state_SHA256']==p['initializer_tensor_SHA256']
    if stage in ('short3','restore_CPU','restore_GPU','final50'):
        capacity=load(inside(p['paths']['capacity_GPU'])/'run_manifest.json')
        assert capacity['passed'] and capacity['status']=='completed_capacity_GPU'
        assert capacity['protocol_SHA256']==sha(protocol_path)
    if stage in ('verify50_CPU','export_probe2_CPU','export548_CPU','metrics_CPU'):
        final=load(inside(p['paths']['final50'])/'run_manifest.json')
        assert final['completed_training_epochs']==50 and final['native_final_eval_returned']
    # Both prepared variants contain the same new RNG synchronization policy.
    # Legacy native11 remains a separate system comparison, not this C0 peer.
    sys.path.insert(0,str(Path(entry).parent))
    os.environ['PYTHONPATH']=os.pathsep.join([str(Path(entry).parent),str(ROOT/'engineering_code'),os.environ.get('PYTHONPATH','')])
    return p,r


def step_deadline(release,started):
    assert time.time()<min(RESULT_END,release['authorized_end_unix'],started+release['max_wall_seconds'])-60


def planning_gate(short_manifest,now,end=RESULT_END,reserve=3600.):
    assert short_manifest['status']=='completed_bounded_h50_short3' and short_manifest['completed_training_epochs']==3
    rows=short_manifest['epochs'];assert [r['epoch'] for r in rows]==[1,2,3]
    times=[rows[i]['native_saved_unix']-rows[i-1]['native_saved_unix'] for i in (1,2)]
    assert all(math.isfinite(v) and v>0 for v in times)
    estimate=max(times)*47*1.25+reserve
    return dict(eligible_for_separate_final_review=now+estimate<=end,actual_epoch2_3_seconds=times,
        estimated_remaining_training_and_source_seal_seconds=estimate,expected_end_unix=now+estimate,
        no_release_or_training_created=True,actual_new_model_final_time_measured=False)
