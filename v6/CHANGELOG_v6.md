# v6 changelog

## Preregistration

* Committed all thresholds, seeds, rival definitions, MCD gate and acceptance rules before executing v6 test lots.
* Fixed `twice_mean` rather than selecting it again from under-resolved C2a folds.

## Comparisons

* Reran Sahodrift and R1 through R5 on simulator seeds 0 through 4, in-prior and shifted, at 24 and 168 hours, at alpha 0.005 and 0.0025.
* Used identical healthy calibration rows and the same deterministic upper-tail conformal rule for every method.
* Added exact intervals, seed-level FAR tails and threshold-resolution fields.

## New evidence

* Evaluated the preregistered within-lot MCD source. It failed the simulator recall gate and was not run on Aabhas.
* Added Aabhas defect-class overlap counts and retained the honest 168 hour lot-MAD win.
* Added BurnTrace results at resolvable 1 percent and 2 percent budgets while retaining 0.5 percent as infeasible.
* Confirmed the 24 hour nominal and 168 hour half-budget policies on fresh seeds 5 through 9.

## Publication

* Updated and redeployed the browser inspector after 18 tests and a production build passed.
* Verified the guided example, performance view and 64-file binary STDF round trip in the deployed application.
* Published the 72 second captioned walkthrough as a release asset rather than committing the binary.
