# SpectroEase verified-extension source snapshot — 2026-10-09

This directory contains the exact source capsule used for the verification-driven extension study. Download and extract **SpectroEase-20261009-source-capsule.zip**, then run commands from the extracted root. The repository's older top-level application and executable are separate versions.

## Integrity and recorded checks

- ZIP SHA-256: `81000e9653f522a722df9521aaa8d024cc67a4c7de23b81e75d55263b4e465ec`.
- 300 ZIP members: 298 selected payload files, a generated empty configuration and an internal CAPSULE_MANIFEST.json with per-file hashes.
- Windows 11, Python 3.12.10; pinned runtime requirements and pytest 8.4.2.
- Source workspace: 469 tests passed, 11 non-failing warnings, 226.26 s; 174 recorded source/test/translation files unchanged.
- Fresh dependency environment and independently extracted capsule: 469 tests passed, 11 non-failing warnings, 245.17 s; capsule members unchanged. Dependency consistency, main import and bounded real MainWindow startup also passed.
- Qt offscreen checks: 12 resolution/scale combinations. This is not certification of physical monitor migration, mixed DPI, or a newly packaged executable.

## Start and reproduce

Install dependencies in a new Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest==8.4.2
python main.py
```

For the exact data-acquisition, split planning, model fitting, accepted-operator replay, independent metric checks and scientific figures, follow `.harness/data/s26/full_workflow_1009/REPRODUCE.md` inside the extracted archive. Use a short writable temporary directory for Windows path-budget tests, as documented there.

## Contents and scope

The snapshot includes application/plugin sources, tests, build specifications, dependency pins, acquisition scripts, frozen protocols, split tables, accepted records, fitted states, predictions, metric reconstructions and figure scripts. Raw source archives, credentials, local runtimes, manuscript files and executables are excluded. Acquire the four primary archives separately and verify the preserved hashes; software licensing does not establish redistribution rights for every raw dataset.

The verified contracts cover manually supplied trusted Python operators for fitted preprocessing, feature selection, classification, regression and scalar metrics. Independent fixtures and explicit acceptance precede execution; held-out rows reuse fitted training state. Six positive implementations, six diagnostic tasks and four small real-data integration subsets support the stated scope. Built-in predictive reference analyses are separate from accepted-operator checks.

Workers enforce bounded execution for trusted code; they are not operating-system security sandboxes. No live LLM-conversion benchmark, R-runtime equivalence, cross-instrument transfer or universal bug-free guarantee is claimed.

The application source is MIT licensed; the original LICENSE.txt is inside the capsule. This is a public research source snapshot provided for review, not an assertion that the older executable contains these changes.
