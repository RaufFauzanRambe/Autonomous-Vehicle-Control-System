# Installation

## Prerequisites

* Python 3.10 or newer (developed and tested on 3.12).
* pip with access to PyPI (or your local wheel mirror).
* No GPU, ROS installation or compiler toolchain is required: every
  module runs on CPU in seconds.

## Dependencies

| Package | Role | Notes |
|---|---|---|
| numpy | arrays, RNG, geometry | >= 1.24 recommended |
| matplotlib | all figures | headless-safe via the Agg backend |
| PyYAML | every `*.yaml` config / scenario | |
| pandas | CSV sample datasets | |
| opencv-python | module 03 perception pipeline | `opencv-python-headless` also works |
| scipy | optional scientific helpers | imported only where used |
| pytest | `10_testing_examples` | >= 7 recommended |

## Install

```bash
# from the repository root
python -m pip install numpy matplotlib PyYAML pandas opencv-python scipy pytest
```

On constrained systems, replace `opencv-python` with
`opencv-python-headless` (identical API, no GUI bindings):

```bash
python -m pip install opencv-python-headless
```

## Verify the installation

```bash
# dependency smoke test
python -c "import numpy, matplotlib, yaml, pandas, cv2, scipy, pytest; print('dependencies OK')"

# one end-to-end module run (writes 01_basic_simulation/outputs/)
python 01_basic_simulation/main.py

# full test suite
python -m pytest 10_testing_examples -v
```

Expected result: the smoke test prints `dependencies OK`, the module run
exits with code 0 and creates `01_basic_simulation/outputs/
basic_simulation_trajectory.png` plus `run_summary.json`, and pytest
reports **19 passed**.

## Repository layout expectations

The suite is executed in-place from the repository root; no packaging
step, virtual environment naming or `PYTHONPATH` export is needed.
Modules locate the project root relative to their own file location, so
the folder can be renamed or moved freely. Namespace packages are used
(there are deliberately no `__init__.py` files), which Python 3.3+
resolves automatically.

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: shared` | the repo was partially copied | ensure `shared/` sits next to the numbered module folders |
| `ImportError: libGL.so.1` | GUI build of OpenCV on a headless server | install `opencv-python-headless` |
| matplotlib hangs or errors on `$DISPLAY` | interactive backend without a display | run without `--show` (defaults to Agg) |
| `yaml.YAMLError` on startup | config file edited with tabs | use spaces; every shipped config is validated |
| pytest collects 0 tests | run from another directory | execute from the repository root |

## Reproducibility

All stochastic modules seed a `numpy.random.default_rng(seed)` from the
`--seed` CLI flag (default 42). Fixing `--seed` makes PNG outputs and
`run_summary.json` byte-identical across runs on the same machine, which
is the baseline for regression checks in experiment scripts.
