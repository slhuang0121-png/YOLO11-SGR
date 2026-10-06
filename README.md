# YOLO11-SGR

Research code accompanying **Tiny Person Detection in UAV Search and Rescue with Selective Geometric Ranking in YOLO11**.

The implementation adds a P2 prediction level to YOLO11m and changes candidate ranking only for augmented-input tiny personnel boxes. Native alignment quality, overlap conflict handling, CIoU and distribution focal loss remain in the retained YOLO11 implementation.

## Study variants

| Variant | Prediction levels | Assignment |
|---|---|---|
| C0 | P3/P4/P5 | Native task-aligned assignment |
| C1 | P3/P4/P5/P2 | Native task-aligned assignment |
| C2 | P3/P4/P5/P2 | Personnel/size-selective geometric ranking, blend 0.25 |

Eligibility: training classes 0/1 and augmented box area <256 input pixels². Evaluation tiny size: square root of the **original** bounding-box area <16 pixels. These two definitions must not be interchanged.

## Installation

Use Python 3.10 in a new environment. Install the matching PyTorch build before this package:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
yolo11-sgr verify-source --repository .
python -m unittest discover -s tests -v
```

For GPU reproduction use the official PyTorch 2.1.2/torchvision 0.16.2 CUDA 12.1 wheels instead of CPU wheels. The experiment used Ultralytics 8.4.170 with the exact eight pinned source files; its retained package is vendored in `src/yolo11_sgr/_vendor`. Do not replace it with the latest pip release. Source hashes are checked by `verify-source`.

## Data and checkpoints

Download VisDrone DET data from its provider. The study used 6,471 training and 548 development images. `Supplementary Data` supplies the exact effective training annotation JSON, original evaluation annotations, predictions, events, FROC arrays and version records. Raw image files are obtained from their respective dataset providers; licensed VCG illustrations are not distributed in this repository.

```sh
yolo11-sgr prepare-labels --annotations /path/to/instances_train_zero_based.json --output /path/to/VisDrone2019-DET-train/labels
yolo11-sgr prepare-labels --annotations /path/to/instances_val_zero_based.json --output /path/to/VisDrone2019-DET-val/labels
```

Copy `configs/visdrone.example.yaml` and set its dataset root. Labels are recreated using the original `.17g` conversion. Original evaluation GT is separate from effective training labels. Class mappings and dataset sources are in [docs/data.md](docs/data.md).

Obtain the official COCO-pretrained `yolo11m.pt` from Ultralytics assets. Required SHA256: `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`. Final selected checkpoint hashes and native-selected epochs are in `reproducibility/model_cards.json`; binary checkpoints are supplied separately with local submission data, not in Git history.

## Training a reproduction

```sh
yolo11-sgr train --variant C2_P2_selective_rank --data /path/to/visdrone.yaml --pretrained /path/to/yolo11m.pt --output /path/to/new_run --device 0 --stop-after 3
yolo11-sgr train --variant C2_P2_selective_rank --data /path/to/visdrone.yaml --resume /path/to/new_run/retained_native_epoch3_before_strip.pt --output /path/to/new_continuation --device 0
```

All recipes retain a cumulative 50-epoch horizon, 1280 input, batch 4, seed 42, AdamW and AMP. `--stop-after 3` saves an unstripped checkpoint without shortening the scheduler horizon. C0/C1 use the corresponding variant names. Native validation selects the best checkpoint before personnel FROC evaluation. The portable launcher is a path-independent wrapper; it was checked without launching another training experiment. Exact historical runners, bounded observations and their original protocols are separately preserved under `reproducibility/frozen_training`; they contain historical paths/time guards and are provenance records, not the recommended user entry point. Historical C1 continuation conditions are documented in Supplementary Methods S1.

## Numerical inference and evaluation

```sh
yolo11-sgr export --checkpoint /path/to/best.pt --images /path/to/images --output /path/to/new_predictions --device cpu
yolo11-sgr evaluate-source --dataset /path/to/VisDrone2019-DET-val --predictions /path/to/new_predictions --output /path/to/new_metrics --repository .
yolo11-sgr reproduce --data /path/to/submission_data --output /path/to/source_policy_points.json
```

The exporter retains full-frame square letterboxing, FP32, confidence floor 0.001, multi-label NMS IoU 0.70 and cap 1000. Four numerical functions are carried across verbatim from the frozen exporter, with AST hashes recorded. `evaluate-source` uses original pixel dimensions, original GT and the verified ignore-region port once. Toolkit cap-500 ten-class AP, COCO101 cap-1000 AP and merged-person FROC are separate metrics.

`reproduce` needs only retained NPZ curves and NumPy: it selects each model's cutoff on the **all-person** curve at 0.10/0.50/1.00 false-positive boxes/image and projects that cutoff unchanged to each of seven strata. No interpolation or equal-score tie splitting is used. It does not train a model or run image inference. Auxiliary Bari/SeaDronesSee cutoffs are fixed on source data, not calibrated on target labels. Their original metric and audit source are retained under `reproducibility/frozen_evaluation`.

## Retained source results

| Model | COCO AP (%) | Tiny recall at primary 0.50 FPI (%) | Tiny recall at secondary 1.00 FPI (%) |
|---|---:|---:|---:|
| C0 | 38.19 | 22.84 | 31.40 |
| C1 | 39.38 | 22.54 | 34.14 |
| C2 | 39.20 | 22.92 | 35.09 |

C2 versus C0 improves COCO AP by 1.02 percentage points and tiny recall by 3.69 points at the retained **secondary 1.00-FPI** policy. Primary tiny change is +0.08 points; primary overall recall changes by −0.52 points. This is a single-seed source development evaluation. The combined C2-versus-C0 effect does not isolate the ranking contribution. Exact counts, cutoffs and all four manuscript tables are retained in `results/` and the numerical supplementary data.

## Repository layout

```text
configs/                 Fixed recipes and example data configuration
src/yolo11_sgr/           Portable CLI and wrappers
  _frozen/               Exact algorithm and evaluator source
  _vendor/ultralytics/    Retained native implementation
reproducibility/         Source pins, parity record, model cards, historical runners
results/                 Manuscript-bound numerical summary and tables
tests/                   Source, assignment and operating-policy checks
docs/                    Data, evaluation and provenance documentation
```

## License and citation

Code is distributed under AGPL-3.0-or-later, consistent with the vendored Ultralytics license. Dataset images and third-party illustrations retain their respective terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). `CITATION.cff` describes this software; no article or archived release DOI has been assigned in this repository.

