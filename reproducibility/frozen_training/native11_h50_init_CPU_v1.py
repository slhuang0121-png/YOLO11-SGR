"""Validate uploaded zero-update initializer; no reconstruction or GPU."""
import argparse,os,shutil,time,traceback
from pathlib import Path
from native11_h50_guard_v1 import ROOT,admit,atomic,sha,tensor_state_sha,step_deadline


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--release',type=Path,required=True);a=ap.parse_args()
    p,r=admit(a.protocol,a.release,'init_CPU',__file__)
    out=ROOT/p['paths']['initialization'];assert not out.exists();out.mkdir(parents=True)
    os.environ['YOLO_CONFIG_DIR']=str(out/'config')
    started=time.time();report=dict(status='running',started_unix=started,protocol_SHA256=sha(a.protocol),
        GPU_used=False,VisDrone_training_epochs=0,optimizer_updates=0,formal_training_released=False)
    path=out/'initialization_proof.json';atomic(path,report)
    try:
        import torch,ultralytics
        from ultralytics.nn.tasks import load_checkpoint
        import model_and_trainer,selective_rank
        torch.set_num_threads(2);assert not torch.cuda.is_available() and ultralytics.__version__=='8.4.170'
        for relative,digest in p['native_source_SHA256'].items():assert sha(Path(ultralytics.__file__).parent/relative)==digest,relative
        original=ROOT/p['initializer_uploaded_relative'];assert sha(original)==p['initializer_file_SHA256']
        model,ck=load_checkpoint(str(original),device='cpu',fuse=False)
        assert ck['epoch']==-1 and ck['optimizer'] is None and ck['updates']==0
        assert tensor_state_sha(model.float())==p['initializer_tensor_SHA256']
        assert model.stride.tolist()==p['expected_strides'] and model.model[-1].nc==10 and model.model[-1].reg_max==16
        checkpoint=out/'zero_VisDrone_updates_prepared_FP32.pt';shutil.copyfile(original,checkpoint)
        step_deadline(r,started)
        report.update(status='completed',passed=True,completed_unix=time.time(),checkpoint=str(checkpoint),
            checkpoint_SHA256=sha(checkpoint),tensor_state_SHA256=tensor_state_sha(model),
            all_initialization_tensor_checks_passed=True,initializer_scope='Prepared C0/P2/selection, exact uploaded zero-update bytes; actual server readback')
        atomic(path,report)
    except Exception:
        report.update(status='failed',passed=False,error=traceback.format_exc());atomic(path,report);raise


if __name__=='__main__':main()
