# v5 changelog

## Calibration protocol

* Recalibrated Fisher, Cauchy, harmonic mean, Tippett and twice-the-mean mergers with the same leave-one-lot-out rule before selection.
* Replaced the small C2b-only final reference with every healthy calibration part.
* Added nominal alpha 0.005 and conservative alpha 0.0025 results.
* Added healthy calibration counts, minimum attainable p values, infeasibility flags, seed-level FAR tails and pooled Clopper-Pearson intervals.

## Review protocol

* Preserved the exact Conformal Risk Control Eq. 4 correction.
* Separated D1, D2 and D4 defects detectable at 24 hours from the all-defect result.
* Added 96 hour and 168 hour review results.

## Evidence and publication

* Added external benchmark rerun code with shared healthy calibration groups and published comparator points.
* Corrected the CORA DOI to `10.34810/data3204` in new v5 materials without rewriting frozen v4 artifacts.
* Deployed the v5 browser inspector to https://sahodrift-app.vercel.app after all 18 supplied tests and a production build passed.
* Added explicit claims that must not be made and exact JSON paths for slide evidence.
