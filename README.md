# CAS Exam 5 — Interactive Study Notebooks

> **Quick start**
>
> ```bash
> git clone https://github.com/cphelan630/exam5.git exam_5
> cd exam_5
> git clone https://github.com/casact/chainladder-python.git chainladder-python
> ```
>
> keep the second repo named `chainladder-python` at the project root; the Python project references it directly.
>
Jupyter notebooks covering the pricing and reserving content of **CAS Exam 5 (Basic Ratemaking and Reserving)**. Each notebook is exam-oriented: it walks through a specific method step-by-step, connects code output back to the CAS formula sheet, and explains the actuarial judgment involved — not just the mechanics.

> **GitHub rendering note**: `.ipynb` files are JSON. GitHub renders a static preview, but outputs are not guaranteed to display correctly and cells must be executed locally to reproduce results. Clone the repo and run the kernel to get the full interactive experience.

---

## Notebooks

### Domain A — Pricing (`notebooks/A01_…A16_`)

| Notebook | Topic |
|---|---|
| [A01](notebooks/A01_insurance_basics_and_ratios.ipynb) | Insurance Basics and Ratios |
| [A02](notebooks/A02_rating_manuals_and_rating_algorithms.ipynb) | Rating Manuals and Rating Algorithms |
| [A03](notebooks/A03_data_and_aggregation.ipynb) | Data and Aggregation |
| [A04](notebooks/A04_exposures_and_exposure_trend.ipynb) | Exposures and Exposure Trend |
| [A05](notebooks/A05_premium_aggregation_onlevel_trend.ipynb) | Premium Aggregation, On-Level, Trend |
| [A06](notebooks/A06_losses_lae_development_trend_adjustments.ipynb) | Losses, LAE, Development and Trend Adjustments |
| [A07](notebooks/A07_expenses_reinsurance_profit_plr.ipynb) | Expenses, Reinsurance, Profit and PLR |
| [A08](notebooks/A08_overall_indication_pp_vs_lr.ipynb) | Overall Indication — Pure Premium vs Loss Ratio |
| [A09](notebooks/A09_univariate_class_ratemaking.ipynb) | Univariate Class Ratemaking |
| [A10](notebooks/A10_multivariate_glm_basics_and_diagnostics.ipynb) | Multivariate GLM Basics and Diagnostics |
| [A11](notebooks/A11_special_classification_territory_ils_deduct_itv_size.ipynb) | Special Classification (Territory, ILFs, Deductibles, ITV, Size) |
| [A12](notebooks/A12_credibility_methods_and_complements.ipynb) | Credibility Methods and Complements |
| [A13](notebooks/A13_other_considerations_reg_ops_market.ipynb) | Other Considerations (Regulatory, Operational, Market) |
| [A14](notebooks/A14_implementation_new_rates_new_product.ipynb) | Implementation — New Rates and New Product |
| [A15](notebooks/A15_commercial_mechanisms_exp_sched_retro_ld.ipynb) | Commercial Mechanisms (Exp. Rating, Schedule Rating, Retro, Loss-Sensitive) |
| [A16](notebooks/A16_claims_made_ratemaking.ipynb) | Claims-Made Ratemaking |

The pricing notebooks use `src/ratingengine/` (installed as a local editable package via `uv sync`) and do not require external data downloads.

### Domain B — Reserving (`notebooks/exam5_*`)

| Notebook | Method(s) Covered |
|---|---|
| [exam5_chainladdermethod.ipynb](notebooks/exam5_chainladdermethod.ipynb) | Development (Chain Ladder) Method |
| [exam5_bf_benktandermethod.ipynb](notebooks/exam5_bf_benktandermethod.ipynb) | Bornhuetter-Ferguson, Benktander, Cape Cod |
| [exam5_expectedclaimsmethod.ipynb](notebooks/exam5_expectedclaimsmethod.ipynb) | Expected Claims Method |
| [exam5_case_outstandingmethod.ipynb](notebooks/exam5_case_outstandingmethod.ipynb) | Case Outstanding Techniques 1 & 2 |
| [exam5_freqsev_disposalmethod.ipynb](notebooks/exam5_freqsev_disposalmethod.ipynb) | Frequency-Severity Techniques 1 & 2, Disposal Rate |
| [exam5_berquistshermanadjustments.ipynb](notebooks/exam5_berquistshermanadjustments.ipynb) | Berquist-Sherman Adjustments |
| [exam5_fullworkflow_methodcomparison.ipynb](notebooks/exam5_fullworkflow_methodcomparison.ipynb) | Full workflow: all methods side-by-side, blending, evaluation |

All reserving notebooks use datasets bundled with `chainladder-python` (Friedland industry auto, XYZ, Berquist-Sherman) — no external data download needed.

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

(These steps are shown again below for reference but you can jump straight to "Install dependencies" if you followed the quick start.)

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

This creates `.venv/` and installs all dependencies (including the local `ratingengine` and `reservingengine` packages from `src/`) into it. The lock file `uv.lock` pins exact versions so the environment is fully reproducible.

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
│   ├── A01_insurance_basics_and_ratios.ipynb     # Domain A pricing notebooks
│   ├── A02_…
│   ├── A16_claims_made_ratemaking.ipynb
│   ├── exam5_chainladdermethod.ipynb             # Domain B reserving notebooks
│   ├── exam5_bf_benktandermethod.ipynb
│   ├── exam5_expectedclaimsmethod.ipynb
│   ├── exam5_case_outstandingmethod.ipynb
│   ├── exam5_freqsev_disposalmethod.ipynb
│   ├── exam5_berquistshermanadjustments.ipynb
│   └── exam5_fullworkflow_methodcomparison.ipynb
├── src/
│   ├── ratingengine/                 # Pricing utilities (Domain A)
│   │   ├── onlevel.py
│   │   ├── trend.py
│   │   ├── indication.py
│   │   ├── loss_adj.py
│   │   ├── expenses.py
│   │   ├── credibility.py
│   │   ├── classratemaking.py
│   │   ├── rating_manual.py
│   │   ├── data_prep.py
│   │   └── types.py
│   └── reservingengine/              # Reserving implementations (Domain B)
│       └── reserving/
│           ├── methods.py
│           ├── diagnostics.py
│           ├── triangle_io.py
│           └── compare.py
├── tests/                            # pytest suite
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
- [`chainladder-python`](https://github.com/casact/chainladder-python) — open-source actuarial library by the CAS
