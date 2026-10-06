"""Independent bounded transfer of the completed matched C0; no legacy restart.

V1 retains original1857 populations/export/matching and uses new C0 complete50
weights/source policy. New finite4h/Oct5-20:00 scope; old Oct4 partial untouched.
Bounded local-CPU transfer of the completed native11 comparator, not training.

Uses its source548 threshold unchanged, complete existing Bari/SDS populations,
and the original frozen numerical exporter and rescue matching functions.
Prepare, four-image probe, full export and metric review remain separate stages.
"""
from pathlib import Path
import argparse, ctypes, hashlib, importlib.util, json, os, sys, time, traceback

R = Path(r"D:\论文\返修")
D = R / "研究短试验/新匹配C0完整50救援1857本机CPU评测20261005_v1"
ARCH = R / "真实实验结果/P2三组完整50原始证据归档20261005_v1/C0_native11/complete_archive/extracted"
RUN = ARCH / "engineering_runs/native11_P2_matched_20261004_v2/C0_native11/final50"
BEST = RUN / "native/weights/best.pt"
SOURCE = ARCH / "engineering_runs/native11_P2_matched_20261004_v2/C0_native11/metrics_CPU/person_FROC7_two_AP.json"
COMPLETE = ARCH / "engineering_runs/native11_P2_matched_20261004_v2/C0_native11/verify50_CPU/completion.json"
GT = {
    "bari": R / "真实实验结果/完整局部复查开发与跨域20261002_v1/datasets/BariSAR_author_6939ab4_flat_v1/instances_all.json",
    "sds": R / "文献研究/官方代码审阅/Ben93kie__SeaDronesSee/OD/evaluation of OD V2/required files/instances_val_iscrowd.json",
}
PINS = {
    str(BEST): "426d6fd2357457cc57e61927b8f21b0c6607cd4c0227e7a3c7dd2a48aedd7dfd",
    str(RUN / "run_manifest.json"): "4cd64e42388c49a4d1d79cf65eee3a112935ac6044d2d7ccfdb756ab3aa6a19d",
    str(COMPLETE): "077ae8ed528194b1e3d83fa42dfe32e5a9792e53f12e75b214a970571e8fd435",
    str(SOURCE): "40c407c984681e864f7ebba125ebb3ebe7d02d6b0cc7fd6e2267fc18d5307010",
    str(GT["bari"]): "80362ad610338f7bf72b1ca80d1f584c24b1200e89b89899830469dc564d521a",
    str(GT["sds"]): "f3175fc0cf5e706b81b273acfad7834bcd4427feb256e634fe5d551b6c004f7e",
    str(R / "评测分析代码/export_yolo_predictions.py"): "bd47863f990926cb51c858a618660091998e7d41c4274b47967f094c79401b02",
    str(R / "评测分析代码/evaluate_paired_h50_rescue_completed50_cpu_v1.py"): "610c42b5b32dd49549498c7be1d2d46d3f4c6bdf7b2059d442da4993703e0604",
    str(R / "局部复查候选代码v2/person_froc_v1.py"): "7445ab1dc13de41ea49d9ce8fe2b9f99b8c9a2fa62e03442839fb2847b87bd3f",
    str(R / "局部复查候选代码v2/box_fusion_v1.py"): "4950918d82144bc28d0b05c894adc7d0a5df788bef0002f8fb4e88e6cb33da46",
    str(R / "局部复查候选代码v2/dataset_selection_adapter_v1.py"): "9f0f37604d2726f7b65f7eb190c7d84e535ecdeaca62bd9a85c65b1c621e7845",
    str(R / "评测分析代码/evaluation_metrics.py"): "87b6b535c33c15b9a6fde358655775c0e392b2b3b7c45a15d6ef69d284c475d6",
    str(R / "局部复查候选代码v2/source_reobservation_v3.py"): "b094bad2a3e95216efe1acc56e761470d9b59db9e054d622f5c6c78e722068a7",
    str(R / "局部复查候选代码v2/seadronessee_person_adapter_v1.py"): "ba15fef303a6f39723f040d1d2f406a9e21525b57fae9dd4bc2f21df8bd220e5",
    str(R / "现代实验代码/visdrone_toolkit_port.py"): "2e0aae0c0d08860e6e291bf9ae1331bd43ea843db037a06d86e11a88b6de403e",
}
FOCI = ("all", "raw_lt16", "raw_lt32", "isolated_lt32")
THRESHOLD = 0.504812657833
DEADLINE = 1791201600.0  # 2026-10-05 20:00 Asia/Shanghai; new independent scope
MAX_EXPORT_WALL_SECONDS = 14400
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="2", NO_ALBUMENTATIONS_UPDATE="1")
sys.stdout.reconfigure(encoding="utf-8")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def save(p, v):
    temp = Path(p).with_suffix(".tmp")
    temp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(p)


def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_pins():
    for p, h in PINS.items():
        assert sha(p) == h, p


def dependency_pins():
    return {str(p): sha(p) for p in sorted((R / "现代实验代码/modern_rescue").rglob("*.py"))}


def memory_available():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    s = Status(); s.length = ctypes.sizeof(s)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.avail_phys


def prepare():
    assert not D.exists(), "Do not repeat prepared population"
    assert "torch" not in sys.modules
    check_pins()
    run, done, src = load(RUN / "run_manifest.json"), load(COMPLETE), load(SOURCE)
    assert run["status"] == "completed_native11_h50_full50" and run["passed"]
    assert run["completed_training_epochs"] == 50 and run["native_final_eval_returned"]
    assert done["status"] == src["status"] == "completed" and done["passed"]
    assert done["checkpoint_SHA256"] == src["checkpoint_SHA256"] == PINS[str(BEST)]
    assert done["selected_checkpoint_actual_epoch"] == src["selected_checkpoint_actual_epoch"] == 46
    point = src["froc"]["all"]["reference_fpi_steps"]["0.5"]
    assert point["threshold"] == THRESHOLD and point["tp"] == 5942 and point["fp"] == 271
    mapping_path = GT["bari"].parent / "mapping.json"
    assert sha(mapping_path) == "b3cd237463644855c22d783e460b0b505deec2731c5715d5ced1067a63815aa1"
    mapping = {r["file_name"]: r for r in load(mapping_path)}
    bbase = R / "真实救援数据准备/自然场景原始来源20261002/bari_author_fixed/uav-search-and-rescue-6939ab477f2fd2850787949c90bee1dbac1fc818/dataset"
    sbase = R / "真实救援数据准备/SeaDronesSee_ODv2/compressed/images/val"
    records = {}; inventories = {}
    for domain in ("bari", "sds"):
        gt = load(GT[domain]); items = []
        assert len(gt["images"]) == (310 if domain == "bari" else 1547)
        assert len(gt["annotations"]) == (547 if domain == "bari" else 9630)
        for im in sorted(gt["images"], key=lambda im: im["file_name"]):
            name = im["file_name"]; assert Path(name).name == name
            if domain == "bari":
                ref = mapping[name]
                image = bbase / ref["scenario"] / "frames" / Path(ref["source_image"]).name
                ann = bbase / ref["scenario"] / "annotations" / Path(ref["source_gt"]).name
                assert sha(image) == ref["image_sha256"] and sha(ann) == ref["annotation_sha256"]
            else:
                image = sbase / name
            digest = sha(image)
            items.append(dict(image=name, image_id=im["id"], path=str(image),
                original_shape=[im["height"], im["width"]], bytes=image.stat().st_size,
                source_image_SHA256=digest))
        inventory = [dict(name=r["image"], bytes=r["bytes"], sha256=r["source_image_SHA256"]) for r in items]
        expected = "7d1eef5143e7d2e4c0400cb1ebae38f3597ee3b0a40593dc9bc3b54b465f16a3" if domain == "bari" else "3b7ca80aa68211165b811a197e8a76fb864a138e922b0f5e9e00a9fb704691d7"
        assert canonical(inventory) == expected, domain
        inventories[domain] = expected; records[domain] = items
    assert memory_available() > 2.5 * 1024**3
    D.mkdir(parents=True)
    save(D / "protocol_before_target_export.json", dict(status="frozen_before_target_export", source_SHA256=sha(__file__),
        created_unix=time.time(), protected_files=PINS, exporter_package_files=dependency_pins(), image_records=records, image_inventories=inventories,
        checkpoint_SHA256=PINS[str(BEST)], source_policy=point, source_FPI_budget=0.5,
        model="new matched C0 native11m complete50 comparator; not legacy native11 or a P2 candidate", completed_epochs=50,
        selected_best_epoch=46, conf=0.001, iou=0.7, max_det=1000, multi_label=True,
        size=1280, head="one2many", precision="fp32", fuse=True, CPU_threads=2, GPU_used=False,
        source_GT_used_only_to_freeze_existing_source_policy=True, target_GT_used_for_selection=False,
        deadline_unix=DEADLINE, max_export_wall_seconds=MAX_EXPORT_WALL_SECONDS, local_available_memory_at_prepare=memory_available(),
        limits=["Existing auxiliary populations already explored; not blind independent disaster missions.",
                "Bari actors/continuous frames, not confirmed real disaster victims.",
                "SDS swimmers only; boats not person; no native five-class AP.",
                "Same geometry/head/NMS as source; local CPU Torch runtime separately recorded.",
                "Source FPI budget need not hold after transfer; retain all negative/null results.",
                "New common-RNG C0 matched comparator; legacy native11 partial is preserved separately."]))
    print(json.dumps(dict(status="prepared", images=1857, source_threshold=THRESHOLD, GPU=False)))


def protocol():
    p = load(D / "protocol_before_target_export.json")
    assert p["source_SHA256"] == sha(__file__) and p["protected_files"] == PINS
    assert p["exporter_package_files"] == dependency_pins()
    check_pins(); assert time.time() < p["deadline_unix"]
    return p


def export(probe):
    p = protocol(); output = D / ("probe4" if probe else "full1857")
    assert not output.exists()
    if not probe:
        q = load(D / "probe4/export_manifest.json")
        assert q["status"] == "completed" and q["images"] == q["exported_images"] == 4
        assert q["source_SHA256"] == sha(__file__) and q["checkpoint_SHA256"] == PINS[str(BEST)]
        assert q["source_threshold"] == THRESHOLD and q["head_reg_max"] == 16
        assert q["planning_required_seconds"] < DEADLINE - time.time(), "Full population cannot fit remaining window"
        assert q["planning_required_seconds"] < MAX_EXPORT_WALL_SECONDS, "Own probe exceeds bounded4h full wall"
    assert memory_available() > 2.5 * 1024**3
    sys.path.insert(0, str(R / "评测分析代码"));sys.path.insert(0, str(R / "现代实验代码"))
    sys.path.insert(0,str(R / "现代实验代码/native11_P2_matched_preparation_20261004_v2"))
    import model_and_trainer, selective_rank
    import cv2, numpy as np, torch, export_yolo_predictions as numerical
    assert sha(numerical.__file__) == PINS[str(R / "评测分析代码/export_yolo_predictions.py")]
    assert not torch.cuda.is_available()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    output.mkdir(); started = time.time()
    report = dict(status="running", started_unix=started, source_SHA256=sha(__file__),
        protocol_SHA256=sha(D / "protocol_before_target_export.json"), checkpoint_SHA256=PINS[str(BEST)],
        purpose="engineering" if probe else "research", source_threshold=THRESHOLD,
        images=4 if probe else 1857, exported_images=0, domains={}, pid=os.getpid(),
        torch_version=torch.__version__, framework_version=numerical.__version__, GPU_used=False,
        model="new matched C0 native11m complete50 comparator", latency_claim=False, training_started=False)
    save(output / "export_manifest.json", report)
    try:
        model, head, info = numerical.load_model(BEST, "one2many", "cpu", 1000, True)
        assert head.nc == 10 and head.reg_max == 16 and not head.end2end
        report.update(info); report["head_reg_max"] = head.reg_max
        times = []
        with torch.inference_mode():
            for domain, population in p["image_records"].items():
                records = population[:2] if probe else population
                dest = output / domain; dest.mkdir()
                corpus = hashlib.sha256(); domain_start = time.time()
                with (dest / "input_prediction_provenance.jsonl").open("x", encoding="utf-8") as stream:
                    for ref in records:
                        assert time.time() < DEADLINE and time.time()-started < (300 if probe else MAX_EXPORT_WALL_SECONDS)
                        assert memory_available() > 750 * 1024**2, "Stop before local memory exhaustion"
                        image = Path(ref["path"]); assert sha(image) == ref["source_image_SHA256"]
                        tick = time.perf_counter(); im = cv2.imread(str(image)); assert im is not None
                        assert list(im.shape[:2]) == ref["original_shape"]
                        tensor, params = numerical.prepare_image(im, 1280)
                        assert tuple(tensor.shape) == (1, 3, 1280, 1280)
                        rows = numerical.prediction_rows(model, head, tensor, params, .001, .7, 1000, True)
                        assert rows.shape[1] == 8 and len(rows) <= 1000 and np.isfinite(rows).all()
                        assert ((rows[:, 5] >= 1) & (rows[:, 5] <= 10)).all()
                        prediction = dest / (Path(ref["image"]).stem + ".txt")
                        assert not prediction.exists(); np.savetxt(prediction, rows, fmt="%.12g", delimiter=",")
                        assert sha(image) == ref["source_image_SHA256"]
                        corpus.update(prediction.name.encode()); corpus.update(b"\0"); corpus.update(prediction.read_bytes()); corpus.update(b"\0")
                        item = ref | dict(prediction_name=prediction.name, prediction_SHA256=sha(prediction),
                            native_prediction_rows=len(rows), person_rows=int(((rows[:, 5] == 1) | (rows[:, 5] == 2)).sum()))
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n"); stream.flush()
                        times.append(time.perf_counter() - tick); report["exported_images"] += 1
                        if report["exported_images"] % 25 == 0:
                            save(output / "export_manifest.json", report)
                            print(json.dumps(dict(exported=report["exported_images"], domain=domain, elapsed=time.time()-started)), flush=True)
                report["domains"][domain] = dict(images=len(records), prediction_corpus_SHA256=corpus.hexdigest(),
                    provenance_SHA256=sha(dest / "input_prediction_provenance.jsonl"), elapsed_seconds=time.time()-domain_start)
                save(output / "export_manifest.json", report)
        assert report["images"] == report["exported_images"]
        check_pins()
        report.update(status="completed", completed_unix=time.time(), actual_sample_image_seconds=times if probe else None,
            planning_required_seconds=(1857 * max(5., max(times)) * 1.10 + 600.) if probe else None,
            all_protected_files_unchanged=True, original_source_threshold_unchanged=True)
        save(output / "export_manifest.json", report)
        print(json.dumps(dict(status="completed", stage="probe4" if probe else "full1857",
            images=report["images"], seconds=time.time()-started, planning_required_seconds=report["planning_required_seconds"])))
    except Exception:
        report.update(status="failed_preserve_partials", error=traceback.format_exc(), failed_unix=time.time())
        save(output / "export_manifest.json", report); raise


def evaluate():
    p = protocol(); exp = D / "full1857"; em = load(exp / "export_manifest.json")
    assert em["status"] == "completed" and em["images"] == em["exported_images"] == 1857
    assert em["source_SHA256"] == sha(__file__) and em["source_threshold"] == THRESHOLD
    assert em["all_protected_files_unchanged"] and em["checkpoint_SHA256"] == PINS[str(BEST)]
    out = D / "metrics"; assert not out.exists();out.mkdir()
    sys.path.insert(0, str(R / "局部复查候选代码v2"));sys.path.insert(0, str(R / "现代实验代码"))
    import numpy as np
    from person_froc_v1 import curve, match_frame
    from source_reobservation_v3 import to_xyxy
    from seadronessee_person_adapter_v1 import prepare_swimmer_frame
    spec = importlib.util.spec_from_file_location("unchanged_rescue_frames", R / "评测分析代码/evaluate_paired_h50_rescue_completed50_cpu_v1.py")
    parent = importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    assert "torch" not in sys.modules
    report = dict(status="running", passed=False, started_unix=time.time(), source_SHA256=sha(__file__),
        protocol_SHA256=sha(D / "protocol_before_target_export.json"), export_manifest_SHA256=sha(exp / "export_manifest.json"),
        checkpoint_SHA256=PINS[str(BEST)], training_completed_epochs=50, source_threshold=THRESHOLD,
        source_metric_SHA256=PINS[str(SOURCE)], GPU_used=False, training_started=False, summaries={},
        target_labels_used_for_model_or_threshold_selection=False, independent_disaster_mission_claim=False)
    result = out / "rescue_FROC_source_fixed.json";save(result, report)
    try:
        for domain in ("bari", "sds"):
            assert time.time() < DEADLINE
            gt = load(GT[domain]); folder = exp / domain
            refs = [json.loads(s) for s in (folder / "input_prediction_provenance.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(refs) == len(gt["images"]) == len(p["image_records"][domain])
            assert sha(folder / "input_prediction_provenance.jsonl") == em["domains"][domain]["provenance_SHA256"]
            assert {r["image"] for r in refs} == {im["file_name"] for im in gt["images"]}
            pred = {}; corpus = hashlib.sha256()
            for ref in refs:
                path = folder / ref["prediction_name"];assert sha(path) == ref["prediction_SHA256"]
                assert sha(ref["path"]) == ref["source_image_SHA256"]
                corpus.update(path.name.encode());corpus.update(b"\0");corpus.update(path.read_bytes());corpus.update(b"\0")
                raw = np.loadtxt(path, delimiter=",", ndmin=2) if path.stat().st_size else np.empty((0, 8))
                assert len(raw) == ref["native_prediction_rows"]
                pred[path.stem] = to_xyxy(raw, "visdrone_10class")
            assert corpus.hexdigest() == em["domains"][domain]["prediction_corpus_SHA256"]
            if domain == "bari":
                frames = parent.bari_frames(gt, pred, np, match_frame); scenarios = ("all", "mountains", "beaches")
            else:
                by_image = {im["id"]: [] for im in gt["images"]}
                for ann in gt["annotations"]:by_image[ann["image_id"]].append(ann)
                frames = {focus: [prepare_swimmer_frame(by_image[im["id"]], pred[Path(im["file_name"]).stem],
                    "visdrone_10class", focus=focus) for im in gt["images"]] for focus in FOCI}
                scenarios = ("all",)
            events = out / (domain + "_per_frame_froc_events.jsonl")
            with events.open("x", encoding="utf-8") as stream:
                for focus, group in frames.items():
                    for im, frame in zip(gt["images"], group):
                        stream.write(json.dumps(dict(focus=focus, image=im["file_name"], image_id=im["id"], frame=frame), allow_nan=False) + "\n")
            report["summaries"][domain] = {}
            for scenario in scenarios:
                report["summaries"][domain][scenario] = {}
                for focus, group in frames.items():
                    subset = group if scenario == "all" else [f for f in group if f["scenario"] == scenario]
                    value = curve(subset, operating_threshold=THRESHOLD); points = value.pop("curve")
                    curvefile = out / (domain + "_" + scenario + "_" + focus + "_steps.npz")
                    np.savez_compressed(curvefile, threshold=np.array([np.inf if s["threshold"] is None else s["threshold"] for s in points]),
                        tp=np.array([s["tp"] for s in points],dtype=np.int64), fp=np.array([s["fp"] for s in points],dtype=np.int64),
                        fp_empty=np.array([s["fp_on_no_evaluable_person_images"] for s in points],dtype=np.int64),
                        images=len(subset), target_count=value["target_count"])
                    value.update(curve_file=curvefile.name, curve_SHA256=sha(curvefile), curve_step_count=len(points))
                    report["summaries"][domain][scenario][focus] = value
                    save(result, report)
            report.setdefault("per_frame_event_SHA256", {})[domain] = sha(events)
            print(json.dumps(dict(domain=domain, all_point=report["summaries"][domain]["all"]["all"]["frozen_threshold_operating_point"])), flush=True)
        assert report["summaries"]["bari"]["all"]["all"]["target_count"] == 547
        assert report["summaries"]["sds"]["all"]["all"]["target_count"] == 6206
        check_pins();report.update(status="completed", passed=True, completed_unix=time.time(), all_protected_files_unchanged=True)
        save(result, report)
        print(json.dumps(dict(status="completed", report_SHA256=sha(result))))
    except Exception:
        report.update(status="failed_preserve_partials", error=traceback.format_exc(), failed_unix=time.time())
        save(result, report);raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--stage", choices=["prepare", "probe4", "full1857", "metrics"], required=True)
    stage = parser.parse_args().stage
    if stage == "prepare":prepare()
    elif stage == "metrics":evaluate()
    else:export(stage == "probe4")
