# exam_5

CAS Exam 5 reserving study project with a chainladder-first refactor. Created using GPT-5.3-Codex. 

## Architecture

- `src/reservingengine/reserving/triangle_io.py`  
  Triangle creation, raw-data loading, exposure alignment, and validation checks.
- `src/reservingengine/reserving/methods.py`  
  - Track A (chainladder-native): Development/Tail/Chainladder/BF/Benktander/CapeCod  
  - Track B (Friedland custom): Case OS and Frequency-Severity
  - Case OS has two implementations:
    - `run_case_outstanding_friedland(...)` (textbook-transparent)
    - `run_case_outstanding_chainladder(...)` (chainladder pipeline)
- `src/reservingengine/reserving/diagnostics.py`  
  Link-ratio tables/heatmaps and Mack-style correlation diagnostics.
- `src/reservingengine/reserving/compare.py`  
  Pattern-level baseline snapshots and reconciliation.

Baseline fixture:
- `tests/fixtures/reserving_exam5_baseline.json`

## Setup

## Fresh clone setup (this repo + chainladder-python)

From a clean machine, clone both repositories so `chainladder-python` is a sibling folder in this project root:

```powershell
git clone <your-exam5-repo-url> exam_5
cd exam_5
git clone https://github.com/casact/chainladder-python.git chainladder-python
```

Important:
- Keep the folder name exactly `chainladder-python`
- Keep it directly under this repo root (`exam_5/chainladder-python`)
- `pyproject.toml` is configured to use that local source path

Then install dependencies:

```powershell
uv sync --group dev --group exam5
```

This project uses local chainladder source:
- `chainladder-python/` (editable via `[tool.uv.sources]`)

If that folder is missing, clone it:

```powershell
git clone https://github.com/casact/chainladder-python.git chainladder-python
```

## Kernel + notebook quickstart

Register kernel:

```powershell
uv run python -m ipykernel install --user --name exam5-chainladder --display-name "Python (exam5-chainladder)"
```

If the kernel does not appear in VS Code right away, run `Developer: Reload Window` (or restart VS Code), then reopen the notebook.

Preferred workflow (VS Code):
- `notebooks/cas-exam5-reserving-methods.ipynb`
- open it directly in VS Code
- select kernel `Python (exam5-chainladder)` from the notebook kernel picker

Optional browser workflow:

```powershell
uv run jupyter lab
```

## Tests and checks

Run tests:

```powershell
uv run pytest -q
```

Run lint:

```powershell
uv run ruff check src tests scripts
```

## Rebuild baseline fixture 

This regenerates the expected reserving outputs for fixed datasets/assumptions (stored as a json) so it can be used by reconciliation tests:
- `tests/fixtures/reserving_exam5_baseline.json`

Use this only when output changes are intentional (for example, changed assumptions, method behavior, or dependency behavior you have validated).

Do **not** rebuild just to make failing tests pass. First confirm whether differences are expected or a regression.

```powershell
$env:PYTHONPATH='src'
uv run python scripts/generate_reserving_exam5_baseline.py
```

## Reconciliation usage

`compare.py` provides:
- `snapshot_method_output(...)`: captures method output in a stable, serializable shape
- `compare_results_to_baseline(...)`: compares AY-level and total metrics with tolerances
- `compare_patterns_to_baseline(...)`: compares selected pattern structures (LDF/CDF) so shape drift is caught
- `build_reconciliation_report(...)`: combines all checks into one report with pass/fail and reason codes

The reconciliation checks:
- selected LDF/CDF patterns
- latest diagonal by AY (and segment)
- ultimate by AY (and segment)
- IBNR by AY (and segment)
- totals

Reason codes in failures are intended to make triage faster (for example pattern mismatch vs tail/exposure alignment differences).
