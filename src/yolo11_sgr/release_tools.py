"""Path-independent wrappers. Frozen numerical and assignment functions remain untouched."""
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def exclusive(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_source(args):
    root = args.repository.resolve()
    manifest = read(root / "reproducibility/source_provenance.json")
    rows = manifest["algorithm_and_evaluator_exact_files"] + manifest["vendor_exact_files"]
    failures = [r["path"] for r in rows if not (root / r["path"]).is_file() or sha(root / r["path"]) != r["SHA256"]]
    if failures:
        raise ValueError(f"Source identity mismatch: {failures}")
    print(json.dumps({"passed": True, "exact_files": len(rows)}))


def point_at(curve, threshold):
    import numpy as np
    indices = np.flatnonzero(curve["threshold"] >= threshold)
    index = int(indices[-1]) if len(indices) else 0
    return {"tau": float(threshold), "TP": int(curve["tp"][index]), "FP": int(curve["fp"][index]),
            "GT": int(curve["target_count"]), "images": int(curve["images"])}


def source_policy(curve, budget):
    import numpy as np
    indices = np.flatnonzero(curve["fp"] <= budget * int(curve["images"]))
    best_tp = curve["tp"][indices].max()
    index = int(indices[curve["tp"][indices] == best_tp][0])
    tau = float(curve["threshold"][index])
    if not math.isfinite(tau):
        raise ValueError("No finite selected source cutoff")
    return tau


def reproduce(args):
    import numpy as np
    root = args.data / "raw_records/source"
    if not root.exists():
        root = args.data / "source"
    results = []
    for variant in ["C0_native11", "C1_append_P2", "C2_P2_selective_rank"]:
        metrics = root / variant / "metrics"
        files = list(metrics.glob("*_person_all_steps.npz"))
        if len(files) != 1:
            raise ValueError(f"Need exactly one all-person curve for {variant}")
        with np.load(files[0], allow_pickle=False) as all_curve:
            thresholds = {b: source_policy(all_curve, b) for b in [0.1, 0.5, 1.0]}
        for path in sorted(metrics.glob("*_person_*_steps.npz")):
            focus = path.stem.split("_person_", 1)[1].removesuffix("_steps")
            with np.load(path, allow_pickle=False) as curve:
                for budget, tau in thresholds.items():
                    results.append({"variant": variant, "focus": focus, "source_all_person_FPI_budget": budget,
                                    "curve_SHA256": sha(path), **point_at(curve, tau)})
    assert len(results) == 63
    for row in results:
        row["recall_percent"] = 100 * row["TP"] / row["GT"] if row["GT"] else None
        row["attained_FPI"] = row["FP"] / row["images"]
    write(args.output, {"policy": "Select on each model all-person curve; project unchanged to every stratum", "rows": results})
    print(json.dumps({"passed": True, "rows": len(results), "output": str(args.output)}))


def prepare_labels(args):
    document = read(args.annotations)
    output = exclusive(args.output)
    by_image = {}
    for annotation in document["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    for image in document["images"]:
        lines = []
        for a in by_image.get(image["id"], []):
            x, y, w, h = a["bbox"]
            width, height = image["width"], image["height"]
            values = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
            assert all(math.isfinite(v) for v in values) and w > 0 and h > 0
            lines.append(" ".join([str(a["category_id"])] + [format(v, ".17g") for v in values]))
        (output / (Path(image["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n" if lines else "", encoding="utf8")
    print(json.dumps({"images": len(document["images"]), "annotations": len(document["annotations"]), "input_SHA256": sha(args.annotations)}))


def export(args):
    from .bootstrap import activate
    activate()
    # Register historical checkpoint class names before deserialisation.
    import model_and_trainer, selective_rank
    import cv2
    import torch
    from .inference_core import load_model, prepare_image, prediction_rows
    output = exclusive(args.output)
    torch.set_num_threads(args.threads)
    model, head, info = load_model(args.checkpoint, "one2many", args.device, 1000, True)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Empty image set or colliding prediction filenames")
    records = []
    with torch.inference_mode():
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable image: {path.name}")
            tensor, params = prepare_image(image, 1280)
            rows = prediction_rows(model, head, tensor.to(args.device), params, 0.001, 0.7, 1000, True)
            target = output / (path.stem + ".txt")
            target.write_text("".join(",".join(format(float(v), ".17g") for v in row) + "\n" for row in rows), encoding="utf8")
            records.append({"image": path.name, "image_SHA256": sha(path), "prediction_SHA256": sha(target), "rows": len(rows)})
            write(output / "export_manifest.json", {"status": "running", "images_completed": len(records), "images_total": len(paths)})
    write(output / "export_manifest.json", {"status": "completed", "checkpoint_SHA256": sha(args.checkpoint),
          "images": len(paths), "imgsz": 1280, "FP32": True, "NMS_IoU": 0.7, "confidence_floor": 0.001,
          "cap": 1000, "multi_label": True, "head": info, "records": records})


def evaluate_source(args):
    from .bootstrap import activate
    activate()
    import numpy as np
    from evaluation_metrics import load_visdrone, verified_author_metrics, coco_metrics
    from visdrone_person_adapter_v1 import prepare_frame
    from person_froc_v1 import curve
    output = exclusive(args.output)
    images, gt, categories, detections, prepared, provenance = load_visdrone(args.dataset, args.predictions)
    results = {"images": len(images), "provenance": provenance, "froc": {}}
    foci = ["all", "raw_lt16", "raw_lt32", "isolated_lt32", "occlusion_0", "occlusion_1", "occlusion_2"]
    def already_filtered(g, d, height, width):
        return g, d
    for focus in foci:
        frames = []
        for image, (official_gt, filtered_dt) in zip(images, prepared):
            raw_gt = official_gt.copy()
            raw_gt[:, 4] = 1 - raw_gt[:, 4]
            frames.append(prepare_frame(raw_gt, filtered_dt, image["height"], image["width"], already_filtered, focus))
        metric = curve(frames)
        points = metric.pop("curve")
        path = output / ("export_person_" + focus + "_steps.npz")
        np.savez_compressed(path, threshold=np.array([np.inf if r["threshold"] is None else r["threshold"] for r in points]),
                            tp=np.array([r["tp"] for r in points], dtype=np.int64), fp=np.array([r["fp"] for r in points], dtype=np.int64),
                            fp_empty=np.array([r["fp_on_no_evaluable_person_images"] for r in points], dtype=np.int64),
                            images=len(images), target_count=metric["target_count"])
        metric.update(curve_file=path.name, curve_SHA256=sha(path), ignore_filter_applied_once=True)
        results["froc"][focus] = metric
        with (output / (focus + "_events.jsonl")).open("w", encoding="utf8") as stream:
            for image, frame in zip(images, frames):
                stream.write(json.dumps({"image": image["file_name"], **frame}, allow_nan=False) + "\n")
    results["published_visdrone_toolkit"] = verified_author_metrics(prepared, args.repository / "reproducibility/visdrone_runtime_parity.json")
    results["coco_cap1000_auxiliary"] = coco_metrics(images, gt, categories, detections, 1000)
    results["status"] = "completed"
    write(output / "person_FROC7_two_AP.json", results)


def train(args):
    from .bootstrap import activate
    activate()
    import copy
    import random
    import shutil
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from model_and_trainer import as_observed_native, append_p2, tensor_sha, MatchedNativeTrainer
    from selective_rank import RankedDetectionModel
    config = read(args.repository / "configs" / (args.variant + ".json"))
    recipe = copy.deepcopy(config["recipe"])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output folder; retained experiments are never overwritten")
    if args.resume:
        model, checkpoint = load_checkpoint(str(args.resume), device="cpu", fuse=False)
        assert checkpoint["epoch"] >= 0 and checkpoint["optimizer"] is not None
        recipe["resume"] = str(args.resume.resolve())
    else:
        if args.pretrained is None or sha(args.pretrained) != config["pretrained_SHA256"]:
            raise ValueError("Supply the matching official yolo11m.pt pretrained file")
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        source = YOLO(str(args.pretrained)).model.float().cpu()
        torch.manual_seed(42)
        settings = get_cfg(overrides=dict(recipe, device=args.device))
        base = DetectionModel(copy.deepcopy(source.yaml), nc=10, verbose=False)
        names = {i: n for i, n in enumerate(["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"])}
        factory = SimpleNamespace(args=settings, data={"names": names, "nc": 10})
        base = DetectionTrainer.set_model_names_for_load(factory, base)
        base.load(source, verbose=False)
        assert tensor_sha(base) == "0e153526905c7895cf5c1eb1f34a6b3070750db033d010cb96e1e9b8ffd9fc8f"
        base.args = settings; base.names = names; base.nc = 10
        base.model[-1].max_det = 1000; base.criterion = None
        model = as_observed_native(base) if args.variant == "C0_native11" else append_p2(base)[0]
        if args.variant == "C2_P2_selective_rank":
            model.__class__ = RankedDetectionModel
            model.rank_strength = 0.25
    recipe.update(data=str(args.data.resolve()), device=args.device, project=str(output.parent), name=output.name, exist_ok=False)
    class PortableTrainer(MatchedNativeTrainer):
        def setup_model(self):
            if args.resume:
                # A preloaded module otherwise bypasses native checkpoint restore.
                return checkpoint
            return super().setup_model()
    trainer = PortableTrainer(overrides=recipe)
    trainer.model = model
    def finite_batch(t):
        assert torch.isfinite(t.loss).all(), "Nonfinite loss: stop; no automatic retry"
    def setup(t):
        assert len(t.train_loader.dataset) == 6471 and len(t.test_loader.dataset) == 548
        assert t.epochs == 50 and t.batch_size == 4 and t.model.stride.tolist() == config["expected_strides"]
    def saved(t):
        assert all(math.isfinite(float(v)) for v in t.metrics.values())
        if t.epoch + 1 == args.stop_after:
            shutil.copyfile(t.last, t.save_dir / f"retained_native_epoch{t.epoch+1}_before_strip.pt")
            if args.stop_after < 50:
                t.stop = True
    trainer.add_callback("on_pretrain_routine_end", setup)
    trainer.add_callback("on_train_batch_end", finite_batch)
    trainer.add_callback("on_model_save", saved)
    trainer.train()
