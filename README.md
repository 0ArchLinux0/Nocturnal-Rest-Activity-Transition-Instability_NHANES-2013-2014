# NHANES 2013-2014 Circadian Sleep Analysis

Actigraphy-based physical modeling of sleep-wake dynamics: Markov transition probability (Night P₀₁), Kramers potential well, and multivariate logistic regression with BMI/PHQ-9.

## Data (NHANES 2013-2014)

XPT data are not committed. The canonical raw-data root is `data/raw/`; legacy root-level scripts still expect project-root files or `nhanes_2013_2014_raw/`. KAN-169 mapped and preserved those scripts without changing their paths or callers; path rewiring is deferred to a caller-aware migration. Do not run the legacy downloader until KAN-170 approves the acquisition set. The existing [CDC NHANES 2013–2014 data page](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2013) remains a source reference.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

## Scripts

| Script | Description |
|--------|-------------|
| `nhanes_physica_physics.py` | Shannon entropy, Markov, spectral gap, EPR |
| `nhanes_physica_ultimate.py` | EPR, time-varying Markov (Day/Night), PCA |
| `nhanes_threshold_robustness.py` | Q1/Q3 threshold robustness |
| `nhanes_logistic_validation.py` | Multivariate logistic (Age, Gender, Night P₀₁) |
| `nhanes_ultimate_logistic.py` | Logistic with BMI, PHQ-9 |
| `nhanes_strict_real_analysis.py` | Strict real-data pipeline (requires BMX_H, DPQ_H) |
