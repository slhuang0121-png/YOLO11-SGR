"""Known P2 reference and observational TAL diagnostics, not a novel algorithm.

The diagnostic path delegates assignment and loss unchanged. It observes input
scale after augmentation; it never labels that scale as original-object scale.
"""
import copy
import hashlib
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss


def tensor_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def append_p2(base):
    """Append P2, preserving original P3/P4/P5 tensors and stride[1] floor."""
    cfg = copy.deepcopy(base.yaml)
    assert cfg['scale'] == 'm' and cfg['head'][-1][0] == [16, 19, 22]
    assert len(cfg['backbone']) == 11 and len(cfg['head']) == 13
    cfg['head'] = cfg['head'][:-1] + [
        [16, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 2], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [128, False]],
        [[16, 19, 22, 25], 1, 'Detect', [10]],
    ]
    cfg['nc'] = 10
    torch.manual_seed(42)
    target = ObservedDetectionModel(cfg, nc=10, verbose=False).float().cpu()
    src, dst = base.state_dict(), target.state_dict()
    mapping = {k: 'model.26.' + k[len('model.23.'):] if k.startswith('model.23.') else k for k in src}
    assert len(mapping) == len(set(mapping.values())) == len(src)
    for k, dest in mapping.items():
        assert dest in dst and dst[dest].shape == src[k].shape, (k, dest)
        dst[dest].copy_(src[k])
    assert all(torch.equal(src[k], dst[v]) for k, v in mapping.items())
    extra = sorted(set(dst) - set(mapping.values()))
    assert extra and all(k.startswith(('model.25.', 'model.26.cv2.3.', 'model.26.cv3.3.')) for k in extra)
    target.args = copy.deepcopy(base.args); target.names = copy.deepcopy(base.names)
    target.nc = 10; target.model[-1].max_det = 1000
    assert target.stride.tolist() == [8., 16., 32., 4.]
    return target, mapping, extra


class AssignmentTap:
    """Retain one detached assignment just long enough to observe it."""
    def __init__(self, native):
        self.native = native
        self.last = None
        self.enabled = False

    def __call__(self, *args, **kwargs):
        result = self.native(*args, **kwargs)
        if self.enabled:
            self.last = (args, result)
        return result

    def __getattr__(self, name):
        native = self.__dict__.get('native')
        if native is None:
            raise AttributeError(name)
        return getattr(native, name)


class ObservedNativeLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = AssignmentTap(self.assigner)
        self.observation_enabled = False
        self.observations = []

    def get_assigned_targets_and_loss(self, preds, batch):
        self.assigner.enabled = self.observation_enabled
        result = super().get_assigned_targets_and_loss(preds, batch)
        if self.observation_enabled:
            self._observe()
        self.assigner.last = None
        return result

    @torch.no_grad()
    def _observe(self):
        args, assigned = self.assigner.last
        scores, pred_boxes, centers, labels, boxes, valid = args
        target_labels, target_boxes, target_scores, foreground, target_idx = assigned
        wh = boxes[..., 2:] - boxes[..., :2]
        rows = []
        for b in range(boxes.shape[0]):
            for g in valid[b, :, 0].nonzero().flatten().tolist():
                cls = int(labels[b, g, 0])
                if cls not in (0, 1):
                    continue
                assigned_mask = foreground[b] & (target_idx[b] == g)
                inds = assigned_mask.nonzero().flatten()
                n = len(inds)
                xy = centers[inds]
                inside = ((xy > boxes[b, g, :2]) & (xy < boxes[b, g, 2:])).all(-1)
                mass = target_scores[b, inds].sum()
                # Pixel xyxy IoU, deliberately not CIoU or an evaluation TP.
                pb, gt = pred_boxes[b, inds], boxes[b, g]
                inter = (torch.minimum(pb[:, 2:], gt[2:]) - torch.maximum(pb[:, :2], gt[:2])).clamp(min=0).prod(-1)
                pa = (pb[:, 2:] - pb[:, :2]).clamp(min=0).prod(-1)
                ga = wh[b, g].prod()
                iou = inter / (pa + ga - inter).clamp(min=1e-9)
                rows.append(dict(batch_image=b, padded_gt_index=g, category=cls,
                    input_width_px=float(wh[b, g, 0]), input_height_px=float(wh[b, g, 1]),
                    input_area_lt256=bool(ga < 256), final_fg_count=n,
                    final_fg_centers_inside_original_input_box=int(inside.sum()),
                    soft_target_score_mass=float(mass),
                    mean_assigned_person_score=float(scores[b, inds, cls].mean()) if n else None,
                    mean_assigned_pixel_IoU=float(iou.mean()) if n else None))
        self.observations.append(dict(scope='Augmented input-scale training observation; not raw-object size or accuracy',
            candidate_floor_px=float(self.assigner.native.stride[1]), rows=rows))


class ObservedDetectionModel(DetectionModel):
    def init_criterion(self):
        assert not self.end2end and self.model[-1].reg_max == 16
        return ObservedNativeLoss(self)


def as_observed_native(base):
    # Same Python structure/tensors. Observation changes no model parameter.
    target = copy.deepcopy(base)
    target.__class__ = ObservedDetectionModel
    target.criterion = None
    assert tensor_sha(target) == tensor_sha(base)
    return target


class MatchedNativeTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_start', synchronize_start_rng)

    def get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is not None and weights.model[-1].nc == self.data['nc'] == 10
        target = copy.deepcopy(weights).float()
        assert isinstance(target, ObservedDetectionModel)
        assert target.stride.tolist() in ([8., 16., 32.], [8., 16., 32., 4.])
        target.criterion = None
        assert tensor_sha(target) == tensor_sha(weights.float())
        return target

    def _build_train_pipeline(self):
        assert getattr(self, '_oom_retries', 0) == 0, 'No silent batch reduction'
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self, epoch):
        assert torch.isfinite(self.loss).all() if torch.is_tensor(self.loss) else all(torch.isfinite(v).all() for v in self.loss.values())
        assert all(torch.isfinite(v).all() for v in self.model.state_dict().values())
        return False


def synchronize_start_rng(trainer):
    # Setup/cache construction consumes different draws. This is an explicit
    # NEW common protocol for C0 and C1, not bitwise reuse of the old C0 run.
    import random
    import numpy as np
    seed = trainer.args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
