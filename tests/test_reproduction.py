import argparse
import tempfile
import unittest
from pathlib import Path
import numpy as np
from yolo11_sgr.release_tools import source_policy, point_at, verify_source

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def test_tied_score_budget_does_not_split_group(self):
        curve = {"threshold": np.array([np.inf, .8, .5]), "tp": np.array([0, 2, 6]),
                 "fp": np.array([0, 1, 3]), "images": np.array(4), "target_count": np.array(10)}
        self.assertEqual(source_policy(curve, .5), .8)
        self.assertEqual(point_at(curve, .8)["TP"], 2)

    def test_projection_uses_original_source_cutoff(self):
        curve = {"threshold": np.array([np.inf, .9, .4]), "tp": np.array([0, 1, 8]),
                 "fp": np.array([0, 3, 9]), "images": np.array(4), "target_count": np.array(10)}
        self.assertEqual(point_at(curve, .8)["FP"], 3)

    def test_immutable_sources(self):
        verify_source(argparse.Namespace(repository=ROOT))


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        from yolo11_sgr.bootstrap import activate
        activate()
        import torch
        self.torch = torch
        from ultralytics.utils.tal import TaskAlignedAssigner
        from selective_rank import SelectiveRankAssigner
        kw = dict(topk=3, num_classes=10, alpha=.5, beta=6., stride=[8., 16., 32., 4.], eps=1e-9)
        self.native = TaskAlignedAssigner(**kw)
        self.zero = SelectiveRankAssigner(**kw, strength=0.)
        self.ranked = SelectiveRankAssigner(**kw, strength=.25)

    def payload(self, label=0, side=8.):
        t = self.torch
        anchors = t.tensor([[8.,8.],[9.,8.],[8.,9.],[20.,20.]])
        scores = t.full((1,4,10), .7)
        boxes = t.tensor([[[4.,4.,12.,12.],[5.,4.,13.,12.],[4.,5.,12.,13.],[18.,18.,22.,22.]]])
        gt = t.tensor([[[8.-side/2,8.-side/2,8.+side/2,8.+side/2]]])
        return scores, boxes, anchors, t.tensor([[[float(label)]]]), gt, t.ones((1,1,1))

    def test_zero_blend_exact_native_assignment(self):
        p = self.payload()
        a, b = self.native(*p), self.zero(*p)
        self.assertTrue(all(self.torch.equal(x,y) for x,y in zip(a,b)))

    def test_nonperson_or_large_box_exact_native_assignment(self):
        for label, side in [(3,8.), (0,32.)]:
            p = self.payload(label,side)
            a, b = self.native(*p), self.ranked(*p)
            self.assertTrue(all(self.torch.equal(x,y) for x,y in zip(a,b)))

    def test_native_alignment_and_overlap_are_returned(self):
        p = self.payload()
        self.native(*p); self.ranked(*p)
        scores, boxes, anchors, labels, gt, mask = p
        a = self.native.get_pos_mask(scores,boxes,labels,gt,anchors,mask)
        b = self.ranked.get_pos_mask(scores,boxes,labels,gt,anchors,mask)
        self.assertTrue(self.torch.equal(a[1],b[1]))
        self.assertTrue(self.torch.equal(a[2],b[2]))


if __name__ == "__main__":
    unittest.main()
