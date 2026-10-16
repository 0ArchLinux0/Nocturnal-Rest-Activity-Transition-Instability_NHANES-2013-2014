#!/usr/bin/env python3
"""Verify KAN-174 displays against canonical KAN-173 result files."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import matplotlib.image as mpimg
import pandas as pd
from sklearn.metrics import roc_auc_score

from make_corrected_tables_figures import (
    ANALYSIS_MANIFEST,
    ASSET_MANIFEST,
    CANONICAL_FILES,
    ROOT,
    SOURCE_COMMIT,
    _format_p,
    _tex_escape,
    build_descriptive_frame,
    build_table1,
)


def _close(actual: float, expected: float, *, tolerance: float = 0.00051) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"Displayed value {actual} does not match source {expected}")


def _table(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)


def _assert_tex_covers_csv(csv_path: str, tex_path: str) -> None:
    table = _table(csv_path)
    tex = (ROOT / tex_path).read_text(encoding="utf-8")
    for value in [*table.columns, *table.to_numpy().ravel().tolist()]:
        if _tex_escape(value) not in tex:
            raise AssertionError(f"LaTeX artifact {tex_path} omits table value {value!r}")


def _verify_primary_table() -> None:
    source = pd.read_csv(CANONICAL_FILES["hc3"]).set_index("model")
    displayed = _table("outputs/tables/table2_primary_corrected.csv").set_index("Model")
    for model in ("Model 1", "Model 2"):
        actual = displayed.loc[model]
        expected = source.loc[model]
        if int(actual["N"]) != int(expected["N"]):
            raise AssertionError(f"{model} N differs from the canonical HC3 result")
        for output_column, source_column in (
            ("P01 beta", "beta_P01"),
            ("HC3 SE", "SE_P01"),
            ("R2", "R2"),
            ("Adjusted R2", "adjusted_R2"),
            ("Effect per +0.01 P01", "P01_effect_per_0_01"),
        ):
            _close(float(actual[output_column]), float(expected[source_column]))
        ci = actual["95% CI"].strip("[]").split(",")
        _close(float(ci[0]), float(expected["CI95_low"]))
        _close(float(ci[1]), float(expected["CI95_high"]))
        if actual["p"] != _format_p(expected["p_P01"]):
            raise AssertionError(f"{model} p-value display differs from canonical HC3 result")


def _verify_survey_table() -> None:
    source = pd.read_csv(CANONICAL_FILES["survey"]).set_index("model")
    displayed = _table("outputs/tables/tableS1_survey_weighted.csv").set_index("Model")
    for model in ("Model 1", "Model 2"):
        actual = displayed.loc[model]
        expected = source.loc[model]
        if int(actual["N"]) != int(expected["N"]):
            raise AssertionError(f"{model} survey N differs from canonical result")
        for output_column, source_column in (
            ("P01 beta", "beta_P01"),
            ("SE", "SE_P01"),
        ):
            _close(float(actual[output_column]), float(expected[source_column]))
        ci = actual["95% CI"].strip("[]").split(",")
        _close(float(ci[0]), float(expected["CI95_low"]))
        _close(float(ci[1]), float(expected["CI95_high"]))
        if actual["p"] != _format_p(expected["p_P01"]):
            raise AssertionError(f"{model} survey p-value differs from canonical result")


def _verify_resolution_table() -> None:
    source = pd.read_csv(CANONICAL_FILES["resolution"]).set_index("resolution_min")
    displayed = _table("outputs/tables/tableS2_resolution_sensitivity.csv").set_index(
        "Resolution (min)"
    )
    for resolution in (5, 10, 30, 60):
        actual = displayed.loc[str(resolution)]
        expected = source.loc[resolution]
        if int(actual["N"]) != int(expected["N"]):
            raise AssertionError(f"{resolution}-minute N differs from canonical result")
        for output_column, source_column in (
            ("P01 mean", "P01_mean"),
            ("P01 SD", "P01_SD"),
            ("P01 beta", "beta"),
            ("HC3 SE", "SE_HC3"),
        ):
            _close(float(actual[output_column]), float(expected[source_column]))
        ci = actual["95% CI"].strip("[]").split(",")
        _close(float(ci[0]), float(expected["CI95_low"]))
        _close(float(ci[1]), float(expected["CI95_high"]))
        if actual["p"] != _format_p(expected["p"]):
            raise AssertionError(f"{resolution}-minute p-value differs from canonical result")


def _verify_table1(analysis_manifest: dict) -> None:
    data, _ = build_descriptive_frame(analysis_manifest)
    expected = build_table1(data).astype(str)
    displayed = _table("outputs/tables/table1_corrected.csv")
    pd.testing.assert_frame_equal(displayed, expected, check_dtype=False)
    flow = pd.read_csv(CANONICAL_FILES["flow"]).set_index("metric")["count"]
    if int(displayed.loc[0, "Overall"]) != int(flow["corrected_cohort_N"]):
        raise AssertionError("Table 1 overall N does not match corrected cohort flow")
    waso = displayed.loc[
        displayed["Characteristic"].str.startswith("WASO_proxy_min")
    ].iloc[0]
    if "Not retained" not in waso["Overall"]:
        raise AssertionError("Table 1 must disclose unavailable participant-level WASO values")


def _verify_roc_and_figures(manifest: dict) -> None:
    roc = pd.read_csv(CANONICAL_FILES["roc"]).set_index("model")
    predictions = pd.read_csv(CANONICAL_FILES["oof"])
    y = pd.to_numeric(predictions["PHQ9_ge_10"], errors="raise").astype(int)
    prediction_columns = {
        "demographics_baseline": "baseline_oof_probability",
        "P01_only": "p01_only_oof_probability",
        "demographics_plus_P01": "combined_oof_probability",
    }
    for model, column in prediction_columns.items():
        calculated = roc_auc_score(y, pd.to_numeric(predictions[column], errors="raise"))
        _close(calculated, roc.loc[model, "AUC_OOF"], tolerance=1e-10)
    delta = float(roc.loc["demographics_plus_P01", "AUC_OOF"] - roc.loc["demographics_baseline", "AUC_OOF"])
    _close(delta, roc.loc["demographics_plus_P01", "delta_AUC_vs_baseline"], tolerance=1e-10)
    pdf_path = ROOT / "outputs/figures/roc_corrected_oof.pdf"
    pdf_text = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    figure_labels = {
        "demographics_baseline": "Demographic baseline",
        "P01_only": "P01 only",
        "demographics_plus_P01": "Demographics + P01",
    }
    for model, label in figure_labels.items():
        summary = roc.loc[model]
        expected = (
            f"{label} | OOF AUC {summary['AUC_OOF']:.3f} "
            f"(95% CI {summary['AUC_CI95_low']:.3f}–{summary['AUC_CI95_high']:.3f})"
        )
        if expected not in pdf_text:
            raise AssertionError(f"ROC PDF legend does not display canonical values: {expected}")
    combined = roc.loc["demographics_plus_P01"]
    delta_text = (
        f"Paired ΔAUC (combined vs baseline) = {delta:.3f} "
        f"(95% CI {combined['delta_CI95_low']:.3f} to {combined['delta_CI95_high']:.3f})"
    )
    if delta_text not in pdf_text:
        raise AssertionError(f"ROC PDF omits canonical paired delta annotation: {delta_text}")

    required_figure_paths = [
        "outputs/figures/roc_corrected_oof.png",
        "outputs/figures/roc_corrected_oof.pdf",
        "outputs/figures/resolution_sensitivity_effect.png",
        "outputs/figures/resolution_sensitivity_effect.pdf",
    ]
    for relative in required_figure_paths:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 1000:
            raise AssertionError(f"Missing or empty figure asset: {relative}")
        if path.suffix == ".png":
            image = mpimg.imread(path)
            if image.shape[0] < 1000 or image.shape[1] < 1000:
                raise AssertionError(f"PNG resolution is too low: {relative} {image.shape}")

    table2_tex = (ROOT / "outputs/tables/table2_primary_corrected.tex").read_text(encoding="utf-8")
    table1_tex = (ROOT / "outputs/tables/table1_corrected.tex").read_text(encoding="utf-8")
    table_s2_tex = (ROOT / "outputs/tables/tableS2_resolution_sensitivity.tex").read_text(encoding="utf-8")
    if "raw 0-1 probability scale" not in table2_tex or "times 0.01" not in table2_tex:
        raise AssertionError("Primary-table footnote omits P01 scaling/effect definition")
    if (
        "Model 1 adjusts for Age, Sex, and BMI" not in table2_tex
        or "Model 2 additionally adjusts for PIR, weekday sleep duration, and WASO" not in table2_tex
        or "standardized on each model's complete-case sample" not in table2_tex
    ):
        raise AssertionError("Primary-table footnote omits model specification or standardization")
    if "self-reported usual weekday sleep duration, not TST" not in table1_tex:
        raise AssertionError("Table 1 mislabels or fails to disambiguate SLD010H")
    if "complete-score denominator" not in table1_tex:
        raise AssertionError("Table 1 omits the PHQ-9 percentage denominator")
    if _tex_escape("WASO_proxy_min is an actigraphy-derived proxy") not in table1_tex:
        raise AssertionError("Table 1 does not label the WASO proxy")
    if "not assumed to be interchangeable" not in table_s2_tex:
        raise AssertionError("Resolution table lacks non-interchangeability caveat")

    if manifest.get("potential_landscape_primary") is not False:
        raise AssertionError("Potential-landscape assets must not be primary")
    generated = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "outputs/tables/table1_corrected.csv",
            "outputs/tables/table2_primary_corrected.csv",
            "outputs/tables/tableS1_survey_weighted.csv",
            "outputs/tables/tableS2_resolution_sensitivity.csv",
        )
    )
    if "Sleep_Medication" in generated or "validated WASO" in generated:
        raise AssertionError("Generated tables contain prohibited sleep-variable claims")
    if "nationally representative" in generated.lower() or "clinically important" in generated.lower():
        raise AssertionError("Generated tables contain an unsupported interpretation")


def verify() -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise AssertionError("Asset manifest source commit is not the accepted KAN-173 commit")
    if manifest.get("analysis_manifest") != ANALYSIS_MANIFEST:
        raise AssertionError("Asset manifest points to the wrong analysis manifest")
    analysis_manifest = json.loads((ROOT / ANALYSIS_MANIFEST).read_text(encoding="utf-8"))
    if analysis_manifest.get("status") != "complete":
        raise AssertionError("KAN-173 manifest is not complete")
    for name, path in CANONICAL_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical KAN-173 {name} input: {path}")

    _verify_table1(analysis_manifest)
    _verify_primary_table()
    _verify_survey_table()
    _verify_resolution_table()
    _verify_roc_and_figures(manifest)

    for artifact in manifest["artifacts"]:
        for source in artifact["source_files"]:
            if not (ROOT / source).is_file():
                raise AssertionError(f"Missing manifest source file: {source}")
        if artifact.get("status") == "skipped_not_appropriate":
            if artifact["artifact"] == "outputs/figures/cohort_flow.png":
                artifact["numbers_verified"] = True
            continue
        path = ROOT / artifact["artifact"]
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty listed asset: {artifact['artifact']}")
        if path.suffix == ".tex":
            csv_path = artifact["artifact"].removesuffix(".tex") + ".csv"
            _assert_tex_covers_csv(csv_path, artifact["artifact"])
        artifact["numbers_verified"] = True

    manifest["verification"] = {
        "status": "PASS",
        "verifier": "scripts/verify_corrected_assets.py",
        "canonical_sources_checked": [str(path.relative_to(ROOT)) for path in CANONICAL_FILES.values()],
    }
    ASSET_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    hc3 = pd.read_csv(CANONICAL_FILES["hc3"]).set_index("model")
    roc = pd.read_csv(CANONICAL_FILES["roc"]).set_index("model")
    print("KAN-174 corrected assets verified against canonical KAN-173 CSV inputs.")
    print(
        "Model 1: "
        f"N={int(hc3.loc['Model 1', 'N'])}, beta={hc3.loc['Model 1', 'beta_P01']:.10f}, "
        f"HC3 CI=[{hc3.loc['Model 1', 'CI95_low']:.10f}, {hc3.loc['Model 1', 'CI95_high']:.10f}]"
    )
    print(
        "Model 2: "
        f"N={int(hc3.loc['Model 2', 'N'])}, beta={hc3.loc['Model 2', 'beta_P01']:.10f}, "
        f"HC3 CI=[{hc3.loc['Model 2', 'CI95_low']:.10f}, {hc3.loc['Model 2', 'CI95_high']:.10f}]"
    )
    print(
        "ROC OOF AUCs: "
        f"baseline={roc.loc['demographics_baseline', 'AUC_OOF']:.10f}, "
        f"P01={roc.loc['P01_only', 'AUC_OOF']:.10f}, "
        f"combined={roc.loc['demographics_plus_P01', 'AUC_OOF']:.10f}, "
        f"delta={roc.loc['demographics_plus_P01', 'delta_AUC_vs_baseline']:.10f}"
    )
    print(f"Verified {len(manifest['artifacts']) - 1} generated assets; cohort flow figure skipped.")


if __name__ == "__main__":
    verify()
