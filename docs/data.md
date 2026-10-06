# Data and class mappings

VisDrone DET: https://github.com/VisDrone/VisDrone-Dataset . Training categories are zero-based: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor. Evaluation/export categories are one-based. Personnel FROC merges categories 1 and 2; ten-class AP retains all categories. Ignore regions are evaluated with the frozen original-toolkit adapter.

SeaDronesSee: https://seadronessee.cs.uni-tuebingen.de/ . Bari mountain/beach annotations and provider references are recorded in Supplementary Data and in the manuscript bibliography. Auxiliary evaluation uses 310 Bari images/547 evaluable people and 1,547 SeaDronesSee images/6,206 swimmer targets. Boat, jetski and safety equipment retain non-person meanings. Tiny personnel counts are 3 and 121 respectively. Original target mappings and labels are retained beside the original predictions.

The effective training JSON represents the exact training population, not the original official evaluation GT. It contains 343,200 effective training annotations and 38,759 validation annotations. `prepare-labels` recreates the normalized labels used by the native YOLO11 study. Do not substitute a fresh raw-annotation conversion with different exclusion or rounding rules.

The six VCG qualitative scenes have no person ground truth. They do not provide independent numerical rescue accuracy. Neither their original images nor derived panels are licensed for redistribution by this code repository.
