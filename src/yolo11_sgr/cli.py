"""Portable entry points around the retained implementation; imports are lazy."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLO11-SGR reproducible research tools")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-source", help="Check immutable source and vendor hashes")
    verify.add_argument("--repository", type=Path, default=Path.cwd())
    reproduce = sub.add_parser("reproduce", help="Recompute source policy points from retained NPZ curves")
    reproduce.add_argument("--data", type=Path, required=True)
    reproduce.add_argument("--output", type=Path, required=True)
    labels = sub.add_parser("prepare-labels", help="Recreate normalized labels from the exact effective annotation JSON")
    labels.add_argument("--annotations", type=Path, required=True)
    labels.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export", help="Run the frozen full-frame numerical prediction exporter")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--images", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--device", default="cpu")
    export.add_argument("--threads", type=int, default=2)
    evaluate = sub.add_parser("evaluate-source", help="Compute seven personnel strata and both AP conventions")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--repository", type=Path, default=Path.cwd())
    train = sub.add_parser("train", help="Train a new C0/C1/C2 reproduction with the fixed 50-epoch horizon")
    train.add_argument("--variant", choices=["C0_native11", "C1_append_P2", "C2_P2_selective_rank"], required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--pretrained", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--repository", type=Path, default=Path.cwd())
    train.add_argument("--device", default="0")
    train.add_argument("--stop-after", type=int, choices=range(1, 51), default=50)
    args = parser.parse_args(argv)
    from . import release_tools
    return getattr(release_tools, args.command.replace("-", "_"))(args)


if __name__ == "__main__":
    main()
