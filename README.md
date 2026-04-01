# NHANES 2013-2014 Circadian Sleep Analysis

Actigraphy-based physical modeling of sleep-wake dynamics: Markov transition probability (Night P₀₁), Kramers potential well, and multivariate logistic regression with BMI/PHQ-9.

## Data (NHANES 2013-2014)

`.xpt` files are not committed (large/binary). Use `download_nhanes_2013_2014.sh` or place files under `nhanes_2013_2014_raw/` (or project root). Typical files: `DEMO_H`, `SLQ_H`, `BMX_H`, `DPQ_H`, `PAXHR_H` (~145MB), `PAXMIN_H` (~8.7GB), etc. See [NHANES 2013–2014 Examination](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2013).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
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
