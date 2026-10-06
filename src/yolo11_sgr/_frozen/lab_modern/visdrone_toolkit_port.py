"""Python translation of the frozen author toolkit, not a COCO metric.

Source: VisDrone/VisDrone2018-DET-toolkit, commit
005445782213e20cb91bc50a597db3dd949e749a. Research-use attribution retained.
Preserves its pixel-integral ignore filtering, continuous VOC AP, global
prediction cap, repeated evalClass weighting, and ignore-GT recall denominator.
Source/runtime parity against MATLAB/Octave is still required before release.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read_rows(path):
    lines = path.read_text().splitlines() if path.exists() else []
    rows = [[float(x.strip()) for x in line.rstrip(",").split(",")] for line in lines if line.strip()]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def drop_ignored_regions(gt, detections, height, width):
    objects = gt[gt[:, 5] != 0]
    regions = np.maximum(1, gt[gt[:, 5] == 0, :4])
    if not len(regions):
        return objects, detections
    raster = np.zeros((height, width), dtype=np.int32)
    for x,y,w,h in regions.astype(int):
        raster[y-1:min(height,y+h), x-1:min(width,x+w)] = 1
    integral = raster.cumsum(0).cumsum(1)

    def keep(rows):
        if not len(rows):
            return rows
        positions = np.maximum(1, np.floor(rows[:, :4] + 0.5)).astype(np.int64)
        x = np.clip(positions[:, 0], 1, width) - 1
        y = np.clip(positions[:, 1], 1, height) - 1
        w,h = positions[:, 2], positions[:, 3]
        right = np.minimum(width - 1, x + w)
        bottom = np.minimum(height - 1, y + h)
        coverage = integral[y,x] + integral[bottom,right] - integral[y,right] - integral[bottom,x]
        return rows[coverage / (h * w) < 0.5]

    return keep(objects), keep(detections)


def overlaps(dt, gt, ignored):
    left = np.maximum(dt[:, None, :2], gt[None, :, :2])
    right = np.minimum(dt[:, None, :2] + dt[:, None, 2:4], gt[None, :, :2] + gt[None, :, 2:4])
    intersection = np.maximum(0, right-left).prod(2)
    da = dt[:, 2:4].prod(1)[:, None]
    ga = gt[:, 2:4].prod(1)[None, :]
    denominator = np.where(ignored[None, :], da, da+ga-intersection)
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)


def match(gt, dt, threshold):
    dt = dt[np.argsort(-dt[:,4], kind="stable")]
    gt = gt[np.argsort(gt[:,4], kind="stable")]
    ignored = gt[:,4] == 1
    matched = np.zeros(len(gt), dtype=bool)
    flags = np.zeros(len(dt), dtype=np.int8)
    oa = overlaps(dt[:,:4], gt[:,:4], ignored)
    for d in range(len(dt)):
        valid = (~ignored) & (~matched) & (oa[d] >= threshold)
        if valid.any():
            best_overlap = oa[d,valid].max()
            g = np.flatnonzero(valid & (oa[d] == best_overlap))[-1]
            matched[g] = True
            flags[d] = 1
        elif (ignored & (oa[d] >= threshold)).any():
            flags[d] = -1
    return dt[:,4], flags, len(gt)


def voc_ap(recall, precision):
    mrec = np.r_[0., recall, 1.]
    mpre = np.r_[0., precision, 0.]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return np.sum((mrec[changed] - mrec[changed-1]) * mpre[changed])


def evaluate(prepared):
    ap = np.zeros((10,10))
    ar = np.zeros((10,10,4))
    class_image_counts = np.zeros(10, dtype=int)
    for cls in range(1,11):
        class_image_counts[cls-1] = sum(np.any(gt[:,5] == cls) for gt,dt in prepared)
        for ti, threshold in enumerate(np.linspace(0.5,0.95,10)):
            for ci, cap in enumerate([1,10,100,500]):
                scores, flags, gt_count = [], [], 0
                for gt, dt in prepared:
                    limited = dt[:cap]
                    s, f, count = match(gt[gt[:,5] == cls,:5], limited[limited[:,5] == cls,:5], threshold)
                    scores.extend(s); flags.extend(f); gt_count += count
                order = np.argsort(-np.asarray(scores), kind="stable")
                flags = np.asarray(flags)[order]
                tp, fp = np.cumsum(flags == 1), np.cumsum(flags == 0)
                recall = tp / max(1,gt_count)
                precision = tp / np.maximum(1,tp+fp)
                ar[cls-1,ti,ci] = recall.max(initial=0)
                if cap == 500:
                    ap[cls-1,ti] = voc_ap(recall,precision)
    denom = class_image_counts.sum()
    if not denom:
        return {"AP": None, "reason": "No evaluated categories present"}
    class_weights = class_image_counts / denom
    avg = lambda values: float(np.sum(values * class_weights))
    return {"AP":avg(ap.mean(1)), "AP50":avg(ap[:,0]), "AP75":avg(ap[:,5]),
        **{f"AR{cap}":avg(ar[:,:,i].mean(1)) for i,cap in enumerate([1,10,100,500])},
        "per_class_AP":ap.mean(1).tolist(), "class_image_counts":class_image_counts.tolist(),
        "metric_units":"fractions (multiply by 100 for percent)",
        "definition":"Published toolkit translation, continuous VOC AP, maxDets=500; not Ultralytics/COCO AP",
        "source_commit":"005445782213e20cb91bc50a597db3dd949e749a",
        "runtime_parity_status":"Not yet checked against MATLAB/Octave execution"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = []
    for ann in sorted((args.dataset / "annotations").glob("*.txt")):
        # An absent prediction file is a failed export, not an empty prediction.
        pred = args.predictions / ann.name
        if not pred.exists():
            raise FileNotFoundError(pred)
        gt, dt = read_rows(ann), read_rows(pred)
        if len(dt) and np.any(dt[:-1,4] < dt[1:,4]):
            raise ValueError("Predictions must be globally sorted by confidence before capping")
        with Image.open(args.dataset / "images" / (ann.stem + ".jpg")) as image:
            width, height = image.size
        gt, dt = drop_ignored_regions(gt, dt, height, width)
        gt[:,4] = 1 - gt[:,4]
        prepared.append((gt,dt))
    result = evaluate(prepared)
    result["images"] = len(prepared)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
