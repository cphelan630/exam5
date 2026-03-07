# CAS Exam 5 — Interactive Study Notebooks

> **Quick start**
>
> ```bash
> git clone https://github.com/cphelan630/exam5.git exam_5
> cd exam_5
> git clone https://github.com/casact/chainladder-python.git chainladder-python
> ```
>
> Keep the second repo named `chainladder-python` at the project root; the Python project references it directly.

Jupyter notebooks covering the pricing and reserving content of **CAS Exam 5 (Basic Ratemaking and Reserving)**. Each notebook is exam-oriented: it walks through a specific method step-by-step, connects code output back to the CAS formula sheet, and explains the actuarial judgment involved — not just the mechanics.

> **GitHub rendering note**: `.ipynb` files are JSON. GitHub renders a static preview, but outputs are not guaranteed to display correctly and cells must be executed locally to reproduce results. Clone the repo and run the kernel to get the full interactive experience.

---

## Notebooks

### Pricing (`notebooks/pricing/`)

Content from Werner & Modlin, organized into three groups:

| Notebook | Topic |
|---|---|
| [exam5_data_and_metrics.ipynb](notebooks/pricing/exam5_data_and_metrics.ipynb) | Data aggregation, exposure bases, key metrics (WM Ch. 1–4) |
| [exam5_traditionalratemaking.ipynb](notebooks/pricing/exam5_traditionalratemaking.ipynb) | Overall indication — pure premium and loss ratio methods (WM Ch. 5–8) |
| [exam5_classratemaking.ipynb](notebooks/pricing/exam5_classratemaking.ipynb) | Class ratemaking — univariate relativities, credibility, special topics (WM Ch. 9–12) |

Pricing notebooks implement all calculations directly in-notebook using `chainladder`, `pandas`, and `numpy`. No local `src/` package is required.

### Reserving (`notebooks/reserving/`)

Content from Friedland, one notebook per method or topic group:

| Notebook | Topic |
|---|---|
| [exam5_chainladdermethod.ipynb](notebooks/reserving/exam5_chainladdermethod.ipynb) | Development (Chain Ladder) Method |
| [exam5_bf_benktandermethod.ipynb](notebooks/reserving/exam5_bf_benktandermethod.ipynb) | Bornhuetter-Ferguson, Benktander, Cape Cod |
| [exam5_expectedclaimsmethod.ipynb](notebooks/reserving/exam5_expectedclaimsmethod.ipynb) | Expected Claims Method |
| [exam5_case_outstandingmethod.ipynb](notebooks/reserving/exam5_case_outstandingmethod.ipynb) | Case Outstanding Techniques 1 & 2 |
| [exam5_freqsev_disposalmethod.ipynb](notebooks/reserving/exam5_freqsev_disposalmethod.ipynb) | Frequency-Severity Techniques 1 & 2, Disposal Rate |
| [exam5_berquistshermanadjustments.ipynb](notebooks/reserving/exam5_berquistshermanadjustments.ipynb) | Berquist-Sherman Adjustments |
| [exam5_triangle_diagnostics_adjustments.ipynb](notebooks/reserving/exam5_triangle_diagnostics_adjustments.ipynb) | Triangle diagnostics: link ratio table, calendar year test, heatmap, trends |
| [exam5_development_estimators.ipynb](notebooks/reserving/exam5_development_estimators.ipynb) | Development pattern selection and estimators |
| [exam5_incremental_additive_and_sample_weights.ipynb](notebooks/reserving/exam5_incremental_additive_and_sample_weights.ipynb) | Incremental additive method and sample weight selection |
| [exam5_full_reserving_analysis.ipynb](notebooks/reserving/exam5_full_reserving_analysis.ipynb) | End-to-end template: all methods, diagnostics, and comparison |
| [exam5_fullworkflow_methodcomparison.ipynb](notebooks/reserving/exam5_fullworkflow_methodcomparison.ipynb) | Method comparison and reconciliation workflow |

Reserving notebooks use datasets bundled with `chainladder-python` (Friedland industry auto, XYZ, Berquist-Sherman) — no external data download needed. Most also import helper functions from `src/reservingengine/`.

### Python Basics (`notebooks/python_basics/`)

Reference notebooks for the Python tools used throughout the study notebooks:

| Notebook | Topic |
|---|---|
| [numpy_intermediate.ipynb](notebooks/python_basics/numpy_intermediate.ipynb) | NumPy arrays, broadcasting, and numerical operations |
| [pandas_intermediate.ipynb](notebooks/python_basics/pandas_intermediate.ipynb) | Pandas DataFrames, indexing, groupby, and reshaping |
| [matplotlib_intermediate.ipynb](notebooks/python_basics/matplotlib_intermediate.ipynb) | Matplotlib/Seaborn plotting patterns |
| [modeling_intro.ipynb](notebooks/python_basics/modeling_intro.ipynb) | Intro to scikit-learn modeling concepts used in reserving |

---

## Setup (see quick start above)

### Prerequisites

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install once on your machine:

  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Clone

This project pins `chainladder-python` as a local editable source dependency. Clone both repos so `chainladder-python` sits as a sibling folder inside the project root:

```bash
git clone https://github.com/cphelan630/exam5.git exam_5
cd exam_5
git clone https://github.com/casact/chainladder-python.git chainladder-python
```

> Keep the folder named exactly `chainladder-python` at the project root — `pyproject.toml` references it by that path.

### Install dependencies

```bash
uv sync --group dev --group exam5
```

This creates `.venv/` and installs all dependencies (including the local `reservingengine` package from `src/`) into it. The lock file `uv.lock` pins exact versions so the environment is fully reproducible.

### Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user --name exam5 --display-name "Python (exam5)"
```

### Open a notebook

**VS Code** (recommended): open any `.ipynb` in `notebooks/`, select `Python (exam5)` from the kernel picker. If the kernel doesn't appear, run `Developer: Reload Window` and try again.

**Browser**:

```bash
uv run jupyter lab
```

---

## Project structure

```
exam_5/
├── notebooks/
│   ├── pricing/
│   │   ├── exam5_data_and_metrics.ipynb
│   │   ├── exam5_traditionalratemaking.ipynb
│   │   └── exam5_classratemaking.ipynb
│   ├── reserving/
│   │   ├── exam5_chainladdermethod.ipynb
│   │   ├── exam5_bf_benktandermethod.ipynb
│   │   ├── exam5_expectedclaimsmethod.ipynb
│   │   ├── exam5_case_outstandingmethod.ipynb
│   │   ├── exam5_freqsev_disposalmethod.ipynb
│   │   ├── exam5_berquistshermanadjustments.ipynb
│   │   ├── exam5_triangle_diagnostics_adjustments.ipynb
│   │   ├── exam5_development_estimators.ipynb
│   │   ├── exam5_incremental_additive_and_sample_weights.ipynb
│   │   ├── exam5_full_reserving_analysis.ipynb
│   │   └── exam5_fullworkflow_methodcomparison.ipynb
│   └── python_basics/
│       ├── numpy_intermediate.ipynb
│       ├── pandas_intermediate.ipynb
│       ├── matplotlib_intermediate.ipynb
│       └── modeling_intro.ipynb
├── src/
│   └── reservingengine/              # Reserving helper library
│       └── reserving/
│           ├── methods.py            # run_chain_ladder, run_bf, run_benktander, etc.
│           ├── diagnostics.py        # link_ratio_table, calendar_year_diagnostic, etc.
│           ├── triangle_io.py        # build_exposure_triangle
│           └── compare.py            # snapshot_method_output, build_reconciliation_report
├── docs/
│   └── reservingengine_functions.md  # Full function reference for src/reservingengine
├── chainladder-python/               # CAS chainladder library (local editable install — not committed)
├── pyproject.toml
└── uv.lock
```

---

## Development

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check src tests
```

---

## Regenerating `uv.lock`

The lock file is committed and should be treated as the source of truth for reproducibility. Only regenerate it when you intentionally update dependencies:

```bash
uv lock
```

To add a new dependency and update the lock in one step:

```bash
uv add <package>            # runtime dependency
uv add --group dev <package> # dev-only dependency
```

---

## References

- [CAS Exam 5 Syllabus and Content Outline](https://www.casact.org/exam/exam-5-basic-techniques-ratemaking-and-estimating-claim-liabilities)
- Friedland, J. *Estimating Unpaid Claims Using Basic Techniques* — CAS study note, primary reserving reference
- Werner, G. & Modlin, C. *Basic Ratemaking* — CAS study note, primary pricing reference
- [`chainladder-python`](https://github.com/casact/chainladder-python) — open-source actuarial library by the CAS
