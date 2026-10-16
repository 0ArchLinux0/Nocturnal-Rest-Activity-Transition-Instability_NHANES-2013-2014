#!/usr/bin/env python3
"""Generate KAN-174 tables and figures from accepted KAN-173 outputs.

This presentation-only script never opens PAXMIN and never refits a model.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "ce9ddafa213ba06306a9d071ae4ecf8ed838c90b"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
ASSET_MANIFEST = ROOT / "outputs" / "tables_figures_manifest.json"
ANALYSIS_MANIFEST = "outputs/models/kan173_run_manifest.json"

MANIFEST_PATH = ROOT / ANALYSIS_MANIFEST
CANONICAL_FILES = {
    "hc3": ROOT / "outputs/models/primary_models_hc3.csv",
    "ordinary": ROOT / "outputs/models/primary_models.csv",
    "survey": ROOT / "outputs/models/survey_weighted_models.csv",
    "resolution": ROOT / "outputs/models/resolution_sensitivity.csv",
    "roc": ROOT / "outputs/models/roc_corrected.csv",
    "oof": ROOT / "outputs/models/roc_corrected_oof_predictions.csv",
    "flow": ROOT / "outputs/models/cohort_flow.csv",
    "historical": ROOT / "outputs/models/p01_primary_old_vs_corrected.csv",
    "p01_cache": ROOT / "outputs/final_cohort_pam_metrics.csv",
}


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _read_xport(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required small NHANES input is missing: {path}")
    import pyreadstat

    frame, _ = pyreadstat.read_xport(str(path), usecols=columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame[columns].copy()
    frame["SEQN"] = _numeric(frame["SEQN"]).astype("Int64")
    return frame.dropna(subset=["SEQN"])


def _load_corrected_cohort(analysis_manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    cohort_path = ROOT / "final_full_cohort.csv"
    source_files = ["final_full_cohort.csv"]
    if cohort_path.is_file():
        cohort = pd.read_csv(cohort_path)
    else:
        # Small protocol and demographic files only. The builder helpers apply the
        # accepted PAXDAYD 2-8 wear qualification; this fallback never opens PAXMIN.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from build_final_full_cohort import (
            MIN_VALID_DAYS,
            load_demo,
            load_dpq_phq9,
            load_paxday_wear_summary,
            load_paxhd,
        )

        raw = ROOT / "data/raw"
        demo = load_demo(raw / "DEMO_H.xpt")
        dpq = load_dpq_phq9(raw / "DPQ_H.xpt")
        paxhd = load_paxhd(raw / "PAXHD_H.xpt")
        wear = load_paxday_wear_summary(raw / "PAXDAY_H.xpt")
        if "PAXSTS" in paxhd.columns:
            paxhd = paxhd[_numeric(paxhd["PAXSTS"]).eq(1)]
        cohort = demo.merge(dpq, on="SEQN", how="inner")
        cohort = cohort.merge(paxhd, on="SEQN", how="inner")
        cohort = cohort.merge(wear, on="SEQN", how="inner")
        cohort = cohort.loc[
            (cohort["n_valid_accel_days"] >= MIN_VALID_DAYS)
            & cohort["has_weekend_valid_day"].eq(1)
        ].copy()
        source_files.extend(
            [
                "data/raw/DEMO_H.xpt",
                "data/raw/DPQ_H.xpt",
                "data/raw/PAXHD_H.xpt",
                "data/raw/PAXDAY_H.xpt",
            ]
        )
    if "SEQN" not in cohort:
        raise ValueError("Corrected cohort input lacks SEQN")
    cohort["SEQN"] = _numeric(cohort["SEQN"]).astype("Int64")
    cohort = cohort.dropna(subset=["SEQN"]).copy()
    cohort["SEQN"] = cohort["SEQN"].astype("int64")
    if cohort["SEQN"].duplicated().any():
        raise ValueError("Corrected cohort input contains duplicate SEQN values")
    if len(cohort) != int(analysis_manifest["corrected_cohort_N"]):
        raise ValueError(
            "Corrected cohort row count does not match KAN-173 manifest: "
            f"{len(cohort)} != {analysis_manifest['corrected_cohort_N']}"
        )
    return cohort, source_files


def build_descriptive_frame(analysis_manifest: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Build only the small participant frame needed for Table 1."""
    cohort, cohort_sources = _load_corrected_cohort(analysis_manifest)
    cohort["Age"] = _numeric(cohort["RIDAGEYR"])
    cohort["Sex_code"] = _numeric(cohort["RIAGENDR"])
    cohort["PHQ9_Score"] = _numeric(cohort["PHQ9_Score"]).clip(0, 27)
    cohort["PHQ9_ge_10"] = np.where(
        cohort["PHQ9_Score"].notna(),
        (cohort["PHQ9_Score"] >= 10).astype(float),
        np.nan,
    )

    raw = ROOT / "data/raw"
    demo = _read_xport(raw / "DEMO_H.xpt", ["SEQN", "INDFMPIR"])
    bmx = _read_xport(raw / "BMX_H.xpt", ["SEQN", "BMXBMI"])
    slq = _read_xport(raw / "SLQ_H.xpt", ["SEQN", "SLD010H"])
    for name, frame in (("DEMO_H", demo), ("BMX_H", bmx), ("SLQ_H", slq)):
        if frame["SEQN"].duplicated().any():
            raise ValueError(f"{name} has duplicate SEQN values")

    data = cohort.merge(demo, on="SEQN", how="left", validate="one_to_one")
    data = data.merge(bmx, on="SEQN", how="left", validate="one_to_one")
    data = data.merge(slq, on="SEQN", how="left", validate="one_to_one")
    data["PIR"] = _numeric(data["INDFMPIR"])
    data["BMI"] = _numeric(data["BMXBMI"])
    data["self_reported_weekday_sleep_hours"] = _numeric(data["SLD010H"])
    data.loc[
        data["self_reported_weekday_sleep_hours"].isin([77, 99]),
        "self_reported_weekday_sleep_hours",
    ] = np.nan
    data["Sex"] = data["Sex_code"].map({1.0: "Male", 2.0: "Female"})

    cache = pd.read_csv(CANONICAL_FILES["p01_cache"])
    required_cache = {
        "SEQN",
        "P01_10m",
        "p01_algorithm_version",
        "paxmin_sha256",
        "method_fingerprint",
    }
    if missing := required_cache - set(cache.columns):
        raise ValueError(f"Corrected P01 cache is missing columns: {sorted(missing)}")
    cache["SEQN"] = _numeric(cache["SEQN"]).astype("Int64")
    if cache["SEQN"].duplicated().any():
        raise ValueError("Corrected P01 cache contains duplicate SEQN values")
    expected_provenance = {
        "p01_algorithm_version": analysis_manifest["p01_algorithm_version"],
        "paxmin_sha256": analysis_manifest["paxmin_sha256"],
        "method_fingerprint": analysis_manifest["method_fingerprint"],
    }
    for column, expected in expected_provenance.items():
        if set(cache[column].dropna().astype(str)) != {str(expected)}:
            raise ValueError(f"Corrected P01 cache provenance mismatch in {column}")
    cache["P01_10m"] = _numeric(cache["P01_10m"])
    if cache["P01_10m"].dropna().lt(0).any() or cache["P01_10m"].dropna().gt(1).any():
        raise ValueError("Corrected P01_10m values must remain on [0,1]")
    data = data.merge(cache[["SEQN", "P01_10m"]], on="SEQN", how="left", validate="one_to_one")
    if data["P01_10m"].notna().sum() != int(analysis_manifest["p01_available_N"]):
        raise ValueError("P01 availability does not match the KAN-173 manifest")

    # KAN-173 retained no participant-level WASO proxy in its committed outputs.
    # Do not derive it from PAXMIN or substitute standardized model covariates.
    data["WASO_proxy_min"] = np.nan
    sources = cohort_sources + [
        "data/raw/DEMO_H.xpt",
        "data/raw/BMX_H.xpt",
        "data/raw/SLQ_H.xpt",
        "outputs/final_cohort_pam_metrics.csv",
        "outputs/models/cohort_flow.csv",
        ANALYSIS_MANIFEST,
    ]
    return data, list(dict.fromkeys(sources))


def _n(frame: pd.DataFrame) -> int:
    return int(len(frame))


def _mean_sd(values: pd.Series, digits: int = 1) -> str:
    values = _numeric(values).dropna()
    if values.empty:
        return "Not available"
    return f"n={len(values)}; {values.mean():.{digits}f} ({values.std(ddof=1):.{digits}f})"


def _category(values: pd.Series, value: str, denominator: int) -> str:
    if denominator == 0:
        return "Not available"
    count = int(values.eq(value).sum())
    return f"{count} ({100 * count / denominator:.1f}%)"


def build_table1(data: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, pd.DataFrame]] = [
        ("Overall", data),
        ("PHQ-9 <10", data.loc[data["PHQ9_ge_10"].eq(0)]),
        ("PHQ-9 >=10", data.loc[data["PHQ9_ge_10"].eq(1)]),
    ]
    rows: list[dict[str, str]] = []
    rows.append(
        {
            "Characteristic": "Participants, N",
            **{name: str(_n(frame)) for name, frame in groups},
        }
    )
    for variable, column, digits in [
        ("Age, years", "Age", 1),
        ("BMI, kg/m^2", "BMI", 1),
        ("Poverty-income ratio", "PIR", 2),
        (
            "Self-reported usual weekday sleep duration, hours",
            "self_reported_weekday_sleep_hours",
            1,
        ),
        ("Corrected P01_10m, probability", "P01_10m", 3),
        ("PHQ9_Score, points", "PHQ9_Score", 1),
    ]:
        rows.append(
            {
                "Characteristic": variable,
                **{name: _mean_sd(frame[column], digits) for name, frame in groups},
            }
        )
    sex_rows = []
    for sex_label in ("Male", "Female"):
        sex_rows.append(
            {
                "Characteristic": f"Sex: {sex_label}, n (%)",
                **{
                    name: _category(frame["Sex"], sex_label, int(frame["Sex"].notna().sum()))
                    for name, frame in groups
                },
            }
        )
    rows[1:1] = sex_rows
    rows.append(
        {
            "Characteristic": "PHQ9_ge_10, n (%)",
            "Overall": _category(
                data["PHQ9_ge_10"], 1.0, int(data["PHQ9_ge_10"].notna().sum())
            ),
            "PHQ-9 <10": "Defines column group",
            "PHQ-9 >=10": "Defines column group",
        }
    )
    rows.append(
        {
            "Characteristic": "WASO_proxy_min, minutes (actigraphy-derived proxy)",
            "Overall": "Not retained in participant-level KAN-173 outputs",
            "PHQ-9 <10": "Not available",
            "PHQ-9 >=10": "Not available",
        }
    )
    return pd.DataFrame(rows, columns=["Characteristic", "Overall", "PHQ-9 <10", "PHQ-9 >=10"])


def _format_p(value: Any) -> str:
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    places = max(0, 2 - math.floor(math.log10(value)))
    return f"{value:.{places}f}"


def _tex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _write_csv_tex(
    frame: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
    footnote: str,
    column_format: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\hline",
        " & ".join(_tex_escape(column) for column in frame.columns) + r" \\",
        r"\hline",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_tex_escape(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{\linewidth}\footnotesize",
            _tex_escape(footnote),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _table1_assets(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    table = build_table1(data)
    footnote = (
        "Continuous variables are mean (SD); categorical variables are n (%). "
        "PHQ-9 group columns include participants with complete scores; the "
        "PHQ9_ge_10 percentage uses that complete-score denominator. "
        "SLD010H is self-reported usual weekday sleep duration, not TST. "
        "WASO_proxy_min is an actigraphy-derived proxy; participant-level proxy "
        "values were not retained in the canonical KAN-173 outputs and were not "
        "recomputed for KAN-174. No baseline significance tests are shown."
    )
    csv_path = TABLE_DIR / "table1_corrected.csv"
    tex_path = TABLE_DIR / "table1_corrected.tex"
    _write_csv_tex(
        table,
        csv_path,
        tex_path,
        "Descriptive characteristics of the corrected analytic cohort",
        "tab:table1_corrected",
        footnote,
        "p{5.1cm}p{3.0cm}p{3.0cm}p{3.0cm}",
    )
    return [str(csv_path.relative_to(ROOT)), str(tex_path.relative_to(ROOT))], [
        "outputs/models/cohort_flow.csv",
        ANALYSIS_MANIFEST,
    ]


def _table2_assets(hc3: pd.DataFrame) -> tuple[list[str], list[str]]:
    ordered = hc3.set_index("model").loc[["Model 1", "Model 2"]].reset_index()
    table = pd.DataFrame(
        {
            "Model": ordered["model"],
            "N": _numeric(ordered["N"]).astype(int),
            "P01 beta": _numeric(ordered["beta_P01"]).map(lambda x: f"{x:.3f}"),
            "HC3 SE": _numeric(ordered["SE_P01"]).map(lambda x: f"{x:.3f}"),
            "95% CI": [
                f"[{lo:.3f}, {hi:.3f}]"
                for lo, hi in zip(_numeric(ordered["CI95_low"]), _numeric(ordered["CI95_high"]))
            ],
            "p": _numeric(ordered["p_P01"]).map(_format_p),
            "R2": _numeric(ordered["R2"]).map(lambda x: f"{x:.3f}"),
            "Adjusted R2": _numeric(ordered["adjusted_R2"]).map(lambda x: f"{x:.3f}"),
            "Effect per +0.01 P01": _numeric(ordered["P01_effect_per_0_01"]).map(
                lambda x: f"{x:.3f}"
            ),
        }
    )
    footnote = (
        "P01 is on the raw 0-1 probability scale; the +0.01 effect is beta times "
        "0.01. Model 1 adjusts for Age, Sex, and BMI; Model 2 additionally adjusts "
        "for PIR, weekday sleep duration, and WASO_proxy_min. Continuous covariates "
        "are standardized on each model's complete-case sample. HC3 "
        "heteroskedasticity-robust standard errors are shown."
    )
    csv_path = TABLE_DIR / "table2_primary_corrected.csv"
    tex_path = TABLE_DIR / "table2_primary_corrected.tex"
    _write_csv_tex(
        table,
        csv_path,
        tex_path,
        "Corrected primary association between P01 and PHQ-9 score",
        "tab:table2_primary_corrected",
        footnote,
        "lrrrrrrrr",
    )
    return [str(csv_path.relative_to(ROOT)), str(tex_path.relative_to(ROOT))], [
        "outputs/models/primary_models_hc3.csv",
        ANALYSIS_MANIFEST,
    ]


def _survey_assets(survey: pd.DataFrame) -> tuple[list[str], list[str]]:
    ordered = survey.set_index("model").loc[["Model 1", "Model 2"]].reset_index()
    table = pd.DataFrame(
        {
            "Model": ordered["model"],
            "N": _numeric(ordered["N"]).astype(int),
            "P01 beta": _numeric(ordered["beta_P01"]).map(lambda x: f"{x:.3f}"),
            "SE": _numeric(ordered["SE_P01"]).map(lambda x: f"{x:.3f}"),
            "95% CI": [
                f"[{lo:.3f}, {hi:.3f}]"
                for lo, hi in zip(_numeric(ordered["CI95_low"]), _numeric(ordered["CI95_high"]))
            ],
            "p": _numeric(ordered["p_P01"]).map(_format_p),
        }
    )
    design_fields = {}
    for field in ("weight", "strata", "PSU"):
        values = ordered[field].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"Expected one survey design value for {field}: {values}")
        design_fields[field] = values[0]
    design_df_values = _numeric(ordered["design_df"]).dropna().unique()
    if len(design_df_values) != 1:
        raise ValueError(f"Expected one survey design df: {design_df_values}")
    footnote = (
        "Survey-weighted sensitivity analysis using "
        f"{design_fields['weight']}, {design_fields['strata']}, and "
        f"{design_fields['PSU']}; design df = {int(design_df_values[0])}."
    )
    csv_path = TABLE_DIR / "tableS1_survey_weighted.csv"
    tex_path = TABLE_DIR / "tableS1_survey_weighted.tex"
    _write_csv_tex(
        table,
        csv_path,
        tex_path,
        "Survey-weighted sensitivity analysis",
        "tab:tableS1_survey_weighted",
        footnote,
        "lrrrrr",
    )
    return [str(csv_path.relative_to(ROOT)), str(tex_path.relative_to(ROOT))], [
        "outputs/models/survey_weighted_models.csv",
        ANALYSIS_MANIFEST,
    ]


def _resolution_table_assets(resolution: pd.DataFrame) -> tuple[list[str], list[str]]:
    ordered = resolution.set_index("resolution_min").loc[[5, 10, 30, 60]].reset_index()
    table = pd.DataFrame(
        {
            "Resolution (min)": _numeric(ordered["resolution_min"]).astype(int),
            "N": _numeric(ordered["N"]).astype(int),
            "P01 mean": _numeric(ordered["P01_mean"]).map(lambda x: f"{x:.3f}"),
            "P01 SD": _numeric(ordered["P01_SD"]).map(lambda x: f"{x:.3f}"),
            "P01 beta": _numeric(ordered["beta"]).map(lambda x: f"{x:.3f}"),
            "HC3 SE": _numeric(ordered["SE_HC3"]).map(lambda x: f"{x:.3f}"),
            "95% CI": [
                f"[{lo:.3f}, {hi:.3f}]"
                for lo, hi in zip(_numeric(ordered["CI95_low"]), _numeric(ordered["CI95_high"]))
            ],
            "p": _numeric(ordered["p"]).map(_format_p),
        }
    )
    csv_path = TABLE_DIR / "tableS2_resolution_sensitivity.csv"
    tex_path = TABLE_DIR / "tableS2_resolution_sensitivity.tex"
    _write_csv_tex(
        table,
        csv_path,
        tex_path,
        "P01 association estimates by temporal resolution",
        "tab:tableS2_resolution_sensitivity",
        "Estimates are shown separately by resolution; the resolutions are not assumed to be interchangeable.",
        "rrrrrrrr",
    )
    return [str(csv_path.relative_to(ROOT)), str(tex_path.relative_to(ROOT))], [
        "outputs/models/resolution_sensitivity.csv",
        ANALYSIS_MANIFEST,
    ]


def _roc_figure(roc: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    model_specs = [
        ("demographics_baseline", "Demographic baseline", "baseline_oof_probability", "#1f77b4"),
        ("P01_only", "P01 only", "p01_only_oof_probability", "#ff7f0e"),
        (
            "demographics_plus_P01",
            "Demographics + P01",
            "combined_oof_probability",
            "#2ca02c",
        ),
    ]
    summary = roc.set_index("model")
    y = _numeric(predictions["PHQ9_ge_10"]).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.7), constrained_layout=True)
    for model, label, probability_column, color in model_specs:
        probabilities = _numeric(predictions[probability_column]).to_numpy()
        fpr, tpr, _ = roc_curve(y, probabilities)
        plotted_auc = float(auc(fpr, tpr))
        reported_auc = float(summary.loc[model, "AUC_OOF"])
        if not np.isclose(plotted_auc, reported_auc, rtol=0, atol=1e-10):
            raise ValueError(f"OOF curve AUC differs from canonical summary for {model}")
        low = float(summary.loc[model, "AUC_CI95_low"])
        high = float(summary.loc[model, "AUC_CI95_high"])
        ax.plot(
            fpr,
            tpr,
            lw=2.1,
            color=color,
            label=f"{label} | OOF AUC {reported_auc:.3f} (95% CI {low:.3f}–{high:.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.55", label="_nolegend_")
    combined = summary.loc["demographics_plus_P01"]
    baseline = summary.loc["demographics_baseline"]
    delta = float(combined["AUC_OOF"] - baseline["AUC_OOF"])
    delta_low = float(combined["delta_CI95_low"])
    delta_high = float(combined["delta_CI95_high"])
    ax.text(
        0.02,
        0.98,
        f"Paired ΔAUC (combined vs baseline) = {delta:.3f} "
        f"(95% CI {delta_low:.3f} to {delta_high:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Cross-validated discrimination for PHQ-9 >=10",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(alpha=0.18)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "roc_corrected_oof.png"
    pdf = FIGURE_DIR / "roc_corrected_oof.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def _resolution_figure(resolution: pd.DataFrame) -> list[str]:
    ordered = resolution.set_index("resolution_min").loc[[5, 10, 30, 60]].reset_index()
    x = _numeric(ordered["resolution_min"]).to_numpy()
    effect = _numeric(ordered["beta"]).to_numpy() * 0.01
    low = _numeric(ordered["CI95_low"]).to_numpy() * 0.01
    high = _numeric(ordered["CI95_high"]).to_numpy() * 0.01
    yerr = np.vstack([effect - low, high - effect])
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.errorbar(
        x,
        effect,
        yerr=yerr,
        fmt="o-",
        color="#245a83",
        ecolor="#245a83",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
    )
    ax.axhline(0, color="0.55", linestyle="--", linewidth=1)
    ax.set_xticks(x, [str(int(value)) for value in x])
    ax.set_xlabel("Temporal resolution (minutes)")
    ax.set_ylabel("PHQ-9 score change per +0.01 P01")
    ax.set_title("P01 effect by temporal resolution")
    ax.grid(axis="y", alpha=0.2)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "resolution_sensitivity_effect.png"
    pdf = FIGURE_DIR / "resolution_sensitivity_effect.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def _asset_records(
    table1: list[str],
    table2: list[str],
    survey: list[str],
    resolution_table: list[str],
    roc_figure: list[str],
    resolution_figure: list[str],
    table1_sources: list[str],
    table2_sources: list[str],
    survey_sources: list[str],
    resolution_sources: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    definitions = [
        (table1, "Corrected analytic cohort description", table1_sources, "primary"),
        (table2, "Primary HC3 regression table", table2_sources, "primary"),
        (survey, "Survey-weighted sensitivity table", survey_sources, "supplement"),
        (resolution_table, "Temporal-resolution sensitivity table", resolution_sources, "supplement"),
        (
            roc_figure,
            "Exploratory OOF ROC figure",
            ["outputs/models/roc_corrected_oof_predictions.csv", "outputs/models/roc_corrected.csv", ANALYSIS_MANIFEST],
            "primary",
        ),
        (
            resolution_figure,
            "Resolution-specific coefficient/effect figure",
            ["outputs/models/resolution_sensitivity.csv", ANALYSIS_MANIFEST],
            "primary",
        ),
    ]
    for artifacts, role, sources, placement in definitions:
        for artifact in artifacts:
            records.append(
                {
                    "artifact": artifact,
                    "role": role,
                    "source_files": sources,
                    "source_commit": SOURCE_COMMIT,
                    "analysis_manifest": ANALYSIS_MANIFEST,
                    "primary_or_supplement": placement,
                    "numbers_verified": False,
                }
            )
    records.append(
        {
            "artifact": "outputs/figures/cohort_flow.png",
            "role": "Optional cohort-flow figure",
            "source_files": ["outputs/models/cohort_flow.csv", ANALYSIS_MANIFEST],
            "source_commit": SOURCE_COMMIT,
            "analysis_manifest": ANALYSIS_MANIFEST,
            "primary_or_supplement": "supplement",
            "numbers_verified": False,
            "status": "skipped_not_appropriate",
            "reason": "The supplied counts are not a strict sequential attrition funnel; no flow diagram was manufactured.",
        }
    )
    return records


def generate() -> dict[str, Any]:
    analysis_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if analysis_manifest.get("status") != "complete":
        raise ValueError("KAN-173 analysis manifest is not complete")
    for key, path in CANONICAL_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical KAN-173 {key} input: {path}")

    hc3 = pd.read_csv(CANONICAL_FILES["hc3"])
    survey = pd.read_csv(CANONICAL_FILES["survey"])
    resolution = pd.read_csv(CANONICAL_FILES["resolution"])
    roc = pd.read_csv(CANONICAL_FILES["roc"])
    predictions = pd.read_csv(CANONICAL_FILES["oof"])
    if analysis_manifest.get("survey_model") != "PASS":
        raise ValueError("KAN-173 survey status is not PASS but survey table was requested")

    data, table1_sources = build_descriptive_frame(analysis_manifest)
    table1, table1_additional = _table1_assets(data)
    table1_sources = list(dict.fromkeys(table1_sources + table1_additional))
    table2, table2_sources = _table2_assets(hc3)
    survey_table, survey_sources = _survey_assets(survey)
    resolution_table, resolution_sources = _resolution_table_assets(resolution)
    roc_figures = _roc_figure(roc, predictions)
    resolution_figures = _resolution_figure(resolution)

    records = _asset_records(
        table1,
        table2,
        survey_table,
        resolution_table,
        roc_figures,
        resolution_figures,
        table1_sources,
        table2_sources,
        survey_sources,
        resolution_sources,
    )
    payload = {
        "source_commit": SOURCE_COMMIT,
        "analysis_manifest": ANALYSIS_MANIFEST,
        "asset_generator": "scripts/make_corrected_tables_figures.py",
        "scientific_results_changed": False,
        "manuscript_changed": False,
        "potential_landscape_primary": False,
        "artifacts": records,
    }
    ASSET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ASSET_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records) - 1} table/figure files; cohort flow figure skipped.")
    print(f"Wrote {ASSET_MANIFEST.relative_to(ROOT)}")
    return payload


if __name__ == "__main__":
    generate()
