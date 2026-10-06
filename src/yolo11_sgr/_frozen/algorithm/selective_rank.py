"""Research candidate: tiny-person ranking, native IoU quality targets.

NWD/hybrid assignment is prior art. This is a bounded adaptation, not a proved
novel algorithm or an accuracy result. Candidate geometry/topk, conflict IoU,
classification quality normalization, CIoU and DFL remain native.
"""
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
from model_and_trainer import ObservedDetectionModel, ObservedNativeLoss, AssignmentTap


class SelectiveRankAssigner(TaskAlignedAssigner):
    def __init__(self, *args, strength=.25, input_tiny_side=16., **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 <= strength <= .25 and input_tiny_side == 16.
        self.rank_strength = strength
        self.input_tiny_side = input_tiny_side

    def get_pos_mask(self, scores, pred_boxes, labels, gt_boxes, anchors, mask_gt):
        if self.rank_strength == 0:
            return super().get_pos_mask(scores, pred_boxes, labels, gt_boxes, anchors, mask_gt)
        eligible_geometry = self.select_candidates_in_gts(anchors, gt_boxes, mask_gt)
        native_align, overlaps = self.get_box_metrics(scores, pred_boxes, labels, gt_boxes, eligible_geometry * mask_gt)
        wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
        tiny = (wh.prod(-1) < self.input_tiny_side**2).unsqueeze(-1)
        person = ((labels == 0) | (labels == 1))
        selected = tiny & person & mask_gt.bool()
        if not selected.any():
            topk = self.select_topk_candidates(native_align, topk_mask=mask_gt.expand(-1,-1,self.topk).bool())
            return topk.mul_(eligible_geometry).mul_(mask_gt.bool()), native_align, overlaps
        pc = (pred_boxes[..., 2:] + pred_boxes[..., :2]) / 2
        pw = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp(min=0)
        classes = labels[...,0].long().clamp(0,self.num_classes-1)
        ranking = native_align.clone()
        # Allocate the extra geometric metric only for eligible rows. Native
        # TAL still owns its full matrix; this does not claim bounded GPU RAM.
        for b in range(self.bs):
            gids = selected[b,:,0].nonzero().flatten()
            if not len(gids):
                continue
            gc = (gt_boxes[b,gids,2:] + gt_boxes[b,gids,:2])/2
            gw = wh[b,gids]
            dist2 = (gc.unsqueeze(1)-pc[b].unsqueeze(0)).square().sum(-1)
            dist2 = dist2 + ((gw.unsqueeze(1)-pw[b].unsqueeze(0))/2).square().sum(-1)
            # Dimensionless, input-object scale. Not the AI-TOD constant 12.7.
            scale = gw.prod(-1).clamp(min=16).sqrt().unsqueeze(-1)
            similarity = torch.exp(-torch.sqrt(dist2.clamp(min=self.eps))/scale)
            class_scores = scores[b,:,classes[b,gids]].T
            blend = (1-self.rank_strength)*overlaps[b,gids] + self.rank_strength*similarity
            metric = class_scores.pow(self.alpha) * blend.pow(self.beta)
            ranking[b,gids] = metric * eligible_geometry[b,gids] * mask_gt[b,gids]
        topk = self.select_topk_candidates(ranking, topk_mask=mask_gt.expand(-1,-1,self.topk).bool())
        positive = topk.mul_(eligible_geometry).mul_(mask_gt.bool())
        # Return NATIVE alignment/overlap: the similarity cannot itself increase
        # the soft class quality target or replace native IoU conflict handling.
        return positive, native_align, overlaps


class RankedDetectionModel(ObservedDetectionModel):
    def init_criterion(self):
        criterion = ObservedNativeLoss(self)
        old = criterion.assigner.native
        criterion.assigner = AssignmentTap(SelectiveRankAssigner(topk=old.topk, num_classes=old.num_classes,
            alpha=old.alpha, beta=old.beta, stride=old.stride, eps=old.eps,
            topk2=old.topk2, strength=self.rank_strength, input_tiny_side=16.))
        return criterion
