from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.reporting import save_table
from traffic_poison.thesis_contract import (
    candidate_contract_flags,
    evaluate_cross_result_standards,
    evaluate_main_result_standards,
    resolve_thesis_contract,
    row_sort_key,
)
from traffic_poison.utils import create_run_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis-ready summary tables from experiment directories.")
    parser.add_argument("--metr-baseline-dir", required=True, help="Stable METR-LA clean baseline directory.")
    parser.add_argument("--metr-poison-dir", required=True, help="METR-LA poisoning directory.")
    parser.add_argument("--defense-dir", required=True, help="Defense evaluation directory for the main poisoning run.")
    parser.add_argument("--cross-dir", help="Optional cross-dataset validation directory.")
    parser.add_argument("--output-dir", help="Optional output directory for the summary tables.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def require_columns(frame: pd.DataFrame, required_columns: list[str], source: Path | str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source} is missing required result columns: {joined}")


def require_json_keys(payload: dict[str, Any], required_keys: list[str], source: Path | str) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{source} is missing required result fields: {joined}")


def require_defense_rows(frame: pd.DataFrame, source: Path | str) -> None:
    require_columns(frame, ["name"], source)
    required_by_name = {
        "zscore_anomaly_screen": ["clean_flag_rate", "poison_flag_rate", "flag_rate_gap"],
        "high_freq_energy_check": ["clean_flag_rate", "poison_flag_rate", "flag_rate_gap"],
        "moving_average_asr_effect": ["asr_before", "asr_after", "asr_gap"],
    }
    for row_name, columns in required_by_name.items():
        row = frame[frame["name"] == row_name]
        if row.empty:
            raise ValueError(f"{source} is missing required defense row: {row_name}")
        require_columns(row, columns, f"{source}:{row_name}")
        if row.iloc[0][columns].isna().any():
            raise ValueError(f"{source}:{row_name} has empty required defense values.")


def candidate_id(frame: pd.DataFrame) -> pd.Series:
    sample_selection_mode = (
        frame["sample_selection_mode"] if "sample_selection_mode" in frame.columns else pd.Series("input_energy", index=frame.index)
    )
    target_weight_mode = frame["target_weight_mode"] if "target_weight_mode" in frame.columns else pd.Series("flat", index=frame.index)
    target_horizon_mode = frame["target_horizon_mode"] if "target_horizon_mode" in frame.columns else pd.Series("all", index=frame.index)
    target_horizon_count = frame["target_horizon_count"] if "target_horizon_count" in frame.columns else pd.Series(3, index=frame.index)
    time_smoothing_kernel = frame["time_smoothing_kernel"] if "time_smoothing_kernel" in frame.columns else pd.Series(3, index=frame.index)
    frequency_smoothing_strength = (
        frame["frequency_smoothing_strength"] if "frequency_smoothing_strength" in frame.columns else pd.Series(0.0, index=frame.index)
    )
    frequency_cutoff_ratio = (
        frame["frequency_cutoff_ratio"] if "frequency_cutoff_ratio" in frame.columns else pd.Series(0.5, index=frame.index)
    )
    frequency_decay = frame["frequency_decay"] if "frequency_decay" in frame.columns else pd.Series(0.35, index=frame.index)
    headroom_error_mix = frame["headroom_error_mix"] if "headroom_error_mix" in frame.columns else pd.Series(0.6, index=frame.index)
    global_shift_fraction = frame["global_shift_fraction"] if "global_shift_fraction" in frame.columns else pd.Series(0.3, index=frame.index)
    tail_focus_multiplier = frame["tail_focus_multiplier"] if "tail_focus_multiplier" in frame.columns else pd.Series(1.6, index=frame.index)
    loss_focus_mode = frame["loss_focus_mode"] if "loss_focus_mode" in frame.columns else pd.Series("uniform", index=frame.index)
    loss_selected_node_weight = (
        frame["loss_selected_node_weight"] if "loss_selected_node_weight" in frame.columns else pd.Series(1.15, index=frame.index)
    )
    loss_tail_horizon_weight = (
        frame["loss_tail_horizon_weight"] if "loss_tail_horizon_weight" in frame.columns else pd.Series(1.75, index=frame.index)
    )
    loss_headroom_boost = (
        frame["loss_headroom_boost"] if "loss_headroom_boost" in frame.columns else pd.Series(0.4, index=frame.index)
    )
    return (
        frame["selection_strategy"].astype(str)
        + "|"
        + frame["window_mode"].astype(str)
        + "|"
        + frame["trigger_steps"].astype(str)
        + "|"
        + frame["trigger_node_count"].astype(str)
        + "|"
        + frame["poison_ratio"].astype(str)
        + "|"
        + frame["sigma_multiplier"].astype(str)
        + "|"
        + frame["target_shift_ratio"].astype(str)
        + "|"
        + sample_selection_mode.astype(str)
        + "|"
        + target_weight_mode.astype(str)
        + "|"
        + target_horizon_mode.astype(str)
        + "|"
        + target_horizon_count.astype(str)
        + "|"
        + time_smoothing_kernel.astype(str)
        + "|"
        + frequency_smoothing_strength.astype(str)
        + "|"
        + frequency_cutoff_ratio.astype(str)
        + "|"
        + frequency_decay.astype(str)
        + "|"
        + headroom_error_mix.astype(str)
        + "|"
        + global_shift_fraction.astype(str)
        + "|"
        + tail_focus_multiplier.astype(str)
        + "|"
        + loss_focus_mode.astype(str)
        + "|"
        + loss_selected_node_weight.astype(str)
        + "|"
        + loss_tail_horizon_weight.astype(str)
        + "|"
        + loss_headroom_boost.astype(str)
    )


def annotate_and_sort_candidate_frame(frame: pd.DataFrame, thesis_contract: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        item = dict(row)
        item.update(candidate_contract_flags(item, thesis_contract))
        rows.append(item)
    rows.sort(key=lambda row: row_sort_key(row, thesis_contract), reverse=True)
    return pd.DataFrame(rows)


def dedupe_candidates(frame: pd.DataFrame, thesis_contract: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    working = ensure_columns(
        working,
        {
            "raw_selected_nodes_tail_horizon_attack_success_rate": working.get("attack_success_rate", 0.0),
            "raw_selected_nodes_target_horizons_attack_success_rate": working.get("attack_success_rate", 0.0),
            "raw_selected_nodes_attack_success_rate": working.get("attack_success_rate", 0.0),
            "raw_global_attack_success_rate": working.get("attack_success_rate", 0.0),
            "raw_selected_nodes_tail_horizon_target_shift_attainment": working.get("target_shift_attainment", 0.0),
            "raw_selected_nodes_target_horizons_target_shift_attainment": working.get("target_shift_attainment", 0.0),
            "raw_selected_nodes_target_shift_attainment": working.get("target_shift_attainment", 0.0),
            "raw_global_target_shift_attainment": working.get("target_shift_attainment", 0.0),
            "sample_selection_mode": "input_energy",
            "target_weight_mode": "flat",
            "headroom_error_mix": 0.6,
            "global_shift_fraction": 0.3,
            "tail_focus_multiplier": 1.6,
            "positive_headroom_rate": 0.0,
            "selected_headroom_mean": 0.0,
            "selected_headroom_score_mean": 0.0,
            "loss_focus_mode": "uniform",
            "loss_selected_node_weight": 1.15,
            "loss_tail_horizon_weight": 1.75,
            "loss_headroom_boost": 0.4,
        },
    )
    working["candidate_id"] = candidate_id(working)
    working = annotate_and_sort_candidate_frame(working, thesis_contract)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in working.to_dict(orient="records"):
        key = str(row["candidate_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    deduped_frame = pd.DataFrame(deduped)
    return deduped_frame.drop(columns=["candidate_id"])


def ensure_columns(frame: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    working = frame.copy()
    for column, default_value in defaults.items():
        if column not in working.columns:
            working[column] = default_value
    return working


def build_baseline_table(metr_baseline_dir: Path, cross_dir: Path | None) -> pd.DataFrame:
    rows = [load_json(metr_baseline_dir / "baseline_summary.json") | load_json(metr_baseline_dir / "stability.json")]
    rows[0]["dataset_name"] = "METR-LA"
    rows[0]["best_MAE"] = rows[0]["best_metrics"]["MAE"]
    rows[0]["best_MAPE"] = rows[0]["best_metrics"]["MAPE"]
    rows[0]["best_RMSE"] = rows[0]["best_metrics"]["RMSE"]
    rows[0].pop("best_metrics", None)
    rows[0].pop("checkpoint_path", None)

    if cross_dir is not None and (cross_dir / "cross_baseline_summary.json").exists():
        cross_summary = load_json(cross_dir / "cross_baseline_summary.json")
        rows.append(cross_summary)

    frame = pd.DataFrame(rows)
    desired_columns = [
        "dataset_name",
        "repeat_count",
        "best_seed",
        "best_MAE",
        "best_MAPE",
        "best_RMSE",
        "mae_mean",
        "mae_min",
        "mae_max",
        "mae_relative_spread",
        "stable_under_5_percent",
    ]
    return frame[[column for column in desired_columns if column in frame.columns]]


def build_candidate_table(poison_dir: Path, thesis_contract: dict[str, Any]) -> pd.DataFrame:
    recheck_path = poison_dir / "recheck_results.csv"
    if recheck_path.exists():
        source_path = recheck_path
        source_frame = load_csv(source_path)
    else:
        source_path = poison_dir / "attack_results.csv"
        source_frame = load_csv(source_path)
    require_columns(
        source_frame,
        [
            "selection_strategy",
            "window_mode",
            "trigger_steps",
            "trigger_node_count",
            "poison_ratio",
            "sigma_multiplier",
            "target_shift_ratio",
            "attack_success_rate",
            "clean_MAE_delta_ratio",
            "raw_selected_nodes_tail_horizon_attack_success_rate",
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate",
            "frequency_energy_shift",
            "mean_z_score",
        ],
        source_path,
    )
    frame = source_frame.copy() if recheck_path.exists() else dedupe_candidates(source_frame, thesis_contract)
    frame = frame.copy()
    frame = ensure_columns(
        frame,
        {
            "attack_success_rate_std": 0.0,
            "clean_MAE_delta_ratio_std": 0.0,
            "target_shift_attainment": 0.0,
            "target_shift_attainment_std": 0.0,
            "mean_prediction_shift_ratio": 0.0,
            "mean_prediction_shift_ratio_std": 0.0,
            "raw_global_attack_success_rate": 0.0,
            "raw_selected_nodes_attack_success_rate": 0.0,
            "raw_selected_nodes_tail_horizon_attack_success_rate": 0.0,
            "raw_selected_nodes_target_horizons_attack_success_rate": 0.0,
            "raw_global_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_shift_attainment": 0.0,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_horizons_target_shift_attainment": 0.0,
            "target_horizon_mode": "all",
            "target_horizon_count": 3,
            "time_smoothing_kernel": 3,
            "frequency_smoothing_strength": 0.0,
            "frequency_cutoff_ratio": 0.5,
            "frequency_decay": 0.35,
            "sample_selection_mode": "input_energy",
            "target_weight_mode": "flat",
            "headroom_error_mix": 0.6,
            "global_shift_fraction": 0.3,
            "tail_focus_multiplier": 1.6,
            "positive_headroom_rate": 0.0,
            "selected_headroom_mean": 0.0,
            "selected_headroom_score_mean": 0.0,
            "loss_focus_mode": "uniform",
            "loss_selected_node_weight": 1.15,
            "loss_tail_horizon_weight": 1.75,
            "loss_headroom_boost": 0.4,
        },
    )
    frame = annotate_and_sort_candidate_frame(frame, thesis_contract)
    desired_columns = [
        "candidate_rank",
        "selection_strategy",
        "window_mode",
        "trigger_steps",
        "trigger_node_count",
        "target_horizon_mode",
        "target_horizon_count",
        "time_smoothing_kernel",
        "frequency_smoothing_strength",
        "frequency_cutoff_ratio",
        "poison_ratio",
        "sigma_multiplier",
        "target_shift_ratio",
        "sample_selection_mode",
        "target_weight_mode",
        "headroom_error_mix",
        "global_shift_fraction",
        "tail_focus_multiplier",
        "loss_focus_mode",
        "loss_selected_node_weight",
        "loss_tail_horizon_weight",
        "loss_headroom_boost",
        "attack_success_rate",
        "attack_success_rate_std",
        "raw_global_attack_success_rate",
        "raw_selected_nodes_attack_success_rate",
        "raw_selected_nodes_tail_horizon_attack_success_rate",
        "raw_selected_nodes_target_horizons_attack_success_rate",
        "clean_MAE_delta_ratio",
        "clean_MAE_delta_ratio_std",
        "target_shift_attainment",
        "target_shift_attainment_std",
        "raw_global_target_shift_attainment",
        "raw_selected_nodes_target_shift_attainment",
        "raw_selected_nodes_tail_horizon_target_shift_attainment",
        "raw_selected_nodes_target_horizons_target_shift_attainment",
        "mean_prediction_shift_ratio",
        "mean_prediction_shift_ratio_std",
        "frequency_energy_shift",
        "mean_z_score",
        "anomaly_rate",
        "local_forecast_error_mean",
        "global_forecast_error_mean",
        "selected_poison_score_mean",
        "positive_headroom_rate",
        "selected_headroom_mean",
        "selected_headroom_score_mean",
        "selected_target_horizon_indices",
        "within_budget",
        "direction_ok",
        "strong_direction_ok",
        "minimum_contract_pass",
        "strong_contract_pass",
        "eligible_for_cross_replay",
    ]
    return frame[[column for column in desired_columns if column in frame.columns]]


def build_family_comparison(poison_dir: Path, thesis_contract: dict[str, Any]) -> pd.DataFrame:
    frame = dedupe_candidates(load_csv(poison_dir / "attack_results.csv"), thesis_contract)
    if frame.empty:
        return frame
    frame = ensure_columns(
        frame,
        {
            "target_shift_attainment": 0.0,
            "frequency_energy_shift": 0.0,
            "mean_z_score": 0.0,
            "anomaly_rate": 0.0,
            "raw_selected_nodes_tail_horizon_attack_success_rate": 0.0,
            "raw_selected_nodes_target_horizons_attack_success_rate": 0.0,
            "raw_selected_nodes_attack_success_rate": 0.0,
            "raw_global_attack_success_rate": 0.0,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_horizons_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_shift_attainment": 0.0,
            "raw_global_target_shift_attainment": 0.0,
            "sample_selection_mode": "input_energy",
            "target_weight_mode": "flat",
            "headroom_error_mix": 0.6,
            "global_shift_fraction": 0.3,
            "tail_focus_multiplier": 1.6,
        },
    )
    frame = frame[frame["within_clean_budget"]].copy()
    grouped = (
        frame.groupby(["selection_strategy", "window_mode"], dropna=False)
        .agg(
            candidate_count=("attack_success_rate", "count"),
            best_attack_success_rate=("attack_success_rate", "max"),
            mean_attack_success_rate=("attack_success_rate", "mean"),
            best_local_attack_success_rate=("raw_selected_nodes_tail_horizon_attack_success_rate", "max"),
            mean_local_attack_success_rate=("raw_selected_nodes_tail_horizon_attack_success_rate", "mean"),
            best_target_horizon_local_attack_success_rate=("raw_selected_nodes_target_horizons_attack_success_rate", "max"),
            mean_target_horizon_local_attack_success_rate=("raw_selected_nodes_target_horizons_attack_success_rate", "mean"),
            mean_clean_MAE_delta_ratio=("clean_MAE_delta_ratio", "mean"),
            mean_target_shift_attainment=("target_shift_attainment", "mean"),
            mean_local_target_shift_attainment=("raw_selected_nodes_tail_horizon_target_shift_attainment", "mean"),
            mean_target_horizon_local_target_shift_attainment=("raw_selected_nodes_target_horizons_target_shift_attainment", "mean"),
            mean_frequency_energy_shift=("frequency_energy_shift", "mean"),
            mean_z_score=("mean_z_score", "mean"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
        )
        .reset_index()
        .sort_values(by=["mean_local_attack_success_rate", "mean_attack_success_rate", "mean_clean_MAE_delta_ratio"], ascending=[False, False, True])
    )
    return grouped


def build_single_axis_comparison(
    poison_dir: Path,
    group_key: str,
    thesis_contract: dict[str, Any],
) -> pd.DataFrame:
    frame = dedupe_candidates(load_csv(poison_dir / "attack_results.csv"), thesis_contract)
    if frame.empty:
        return frame
    frame = ensure_columns(
        frame,
        {
            "target_shift_attainment": 0.0,
            "frequency_energy_shift": 0.0,
            "mean_z_score": 0.0,
            "anomaly_rate": 0.0,
            "raw_selected_nodes_tail_horizon_attack_success_rate": 0.0,
            "raw_selected_nodes_target_horizons_attack_success_rate": 0.0,
            "raw_selected_nodes_attack_success_rate": 0.0,
            "raw_global_attack_success_rate": 0.0,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_horizons_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_shift_attainment": 0.0,
            "raw_global_target_shift_attainment": 0.0,
            "selection_strategy": "unknown",
            "window_mode": "unknown",
            "target_horizon_mode": "all",
            "frequency_smoothing_strength": 0.0,
            "headroom_error_mix": 0.6,
            "global_shift_fraction": 0.3,
            "tail_focus_multiplier": 1.6,
        },
    )
    frame = frame[frame["within_clean_budget"]].copy()
    grouped = (
        frame.groupby([group_key], dropna=False)
        .agg(
            candidate_count=("attack_success_rate", "count"),
            mean_attack_success_rate=("attack_success_rate", "mean"),
            best_attack_success_rate=("attack_success_rate", "max"),
            mean_local_attack_success_rate=("raw_selected_nodes_tail_horizon_attack_success_rate", "mean"),
            best_local_attack_success_rate=("raw_selected_nodes_tail_horizon_attack_success_rate", "max"),
            mean_target_horizon_local_attack_success_rate=("raw_selected_nodes_target_horizons_attack_success_rate", "mean"),
            best_target_horizon_local_attack_success_rate=("raw_selected_nodes_target_horizons_attack_success_rate", "max"),
            mean_clean_MAE_delta_ratio=("clean_MAE_delta_ratio", "mean"),
            mean_target_shift_attainment=("target_shift_attainment", "mean"),
            mean_local_target_shift_attainment=("raw_selected_nodes_tail_horizon_target_shift_attainment", "mean"),
            mean_target_horizon_local_target_shift_attainment=("raw_selected_nodes_target_horizons_target_shift_attainment", "mean"),
            mean_frequency_energy_shift=("frequency_energy_shift", "mean"),
            mean_z_score=("mean_z_score", "mean"),
            mean_anomaly_rate=("anomaly_rate", "mean"),
        )
        .reset_index()
        .sort_values(by=["mean_local_attack_success_rate", "mean_attack_success_rate", "best_local_attack_success_rate"], ascending=[False, False, False])
    )
    return grouped


def build_parameter_sensitivity(poison_dir: Path, thesis_contract: dict[str, Any]) -> pd.DataFrame:
    frame = dedupe_candidates(load_csv(poison_dir / "attack_results.csv"), thesis_contract)
    if frame.empty:
        return frame
    frame = ensure_columns(
        frame,
        {
            "target_shift_attainment": 0.0,
            "mean_prediction_shift_ratio": 0.0,
            "frequency_energy_shift": 0.0,
            "mean_z_score": 0.0,
            "anomaly_rate": 0.0,
            "raw_global_attack_success_rate": 0.0,
            "raw_selected_nodes_attack_success_rate": 0.0,
            "raw_selected_nodes_tail_horizon_attack_success_rate": 0.0,
            "raw_global_target_shift_attainment": 0.0,
            "raw_selected_nodes_target_shift_attainment": 0.0,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": 0.0,
            "target_horizon_mode": "all",
            "target_horizon_count": 3,
            "time_smoothing_kernel": 3,
            "frequency_smoothing_strength": 0.0,
            "frequency_cutoff_ratio": 0.5,
            "sample_selection_mode": "input_energy",
            "target_weight_mode": "flat",
            "headroom_error_mix": 0.6,
            "global_shift_fraction": 0.3,
            "tail_focus_multiplier": 1.6,
        },
    )
    frame = frame[frame["within_clean_budget"]].copy()
    frame = annotate_and_sort_candidate_frame(frame, thesis_contract)
    desired_columns = [
        "selection_strategy",
        "window_mode",
        "trigger_steps",
        "trigger_node_count",
        "target_horizon_mode",
        "target_horizon_count",
        "time_smoothing_kernel",
        "frequency_smoothing_strength",
        "frequency_cutoff_ratio",
        "frequency_decay",
        "poison_ratio",
        "sigma_multiplier",
        "target_shift_ratio",
        "sample_selection_mode",
        "target_weight_mode",
        "attack_success_rate",
        "raw_global_attack_success_rate",
        "raw_selected_nodes_attack_success_rate",
        "raw_selected_nodes_tail_horizon_attack_success_rate",
        "clean_MAE_delta_ratio",
        "target_shift_attainment",
        "raw_global_target_shift_attainment",
        "raw_selected_nodes_target_shift_attainment",
        "raw_selected_nodes_tail_horizon_target_shift_attainment",
        "mean_prediction_shift_ratio",
        "frequency_energy_shift",
        "mean_z_score",
        "anomaly_rate",
        "local_forecast_error_mean",
        "global_forecast_error_mean",
        "selected_poison_score_mean",
        "selected_time_indices",
        "selected_target_horizon_indices",
        "selected_nodes",
    ]
    return frame[[column for column in desired_columns if column in frame.columns]]


def build_defense_table(defense_dir: Path) -> pd.DataFrame:
    summary_path = defense_dir / "defense_summary.json"
    summary = load_json(summary_path)
    require_json_keys(summary, ["rows"], summary_path)
    frame = pd.DataFrame(summary.get("rows", []))
    if frame.empty:
        raise ValueError(f"{summary_path} does not contain any defense rows.")
    if "clean_flag_rate" in frame.columns and "poison_flag_rate" in frame.columns and "flag_rate_gap" not in frame.columns:
        frame["flag_rate_gap"] = frame["poison_flag_rate"] - frame["clean_flag_rate"]
    if "asr_before" in frame.columns and "asr_after" in frame.columns and "asr_gap" not in frame.columns:
        frame["asr_gap"] = frame["asr_after"] - frame["asr_before"]
    require_defense_rows(frame, summary_path)
    return frame


def build_summary_payload(
    baseline_table: pd.DataFrame,
    poison_dir: Path,
    cross_dir: Path | None,
    candidate_table: pd.DataFrame,
    strategy_table: pd.DataFrame,
    window_table: pd.DataFrame,
    thesis_contract: dict[str, Any],
) -> dict[str, Any]:
    search_summary = load_json(poison_dir / "search_summary.json")
    final_best = search_summary.get("final_best_paper", search_summary.get("final_best", {}))
    raw_best = search_summary.get("final_best_raw", {})
    baseline_best_mae = float(baseline_table.iloc[0]["best_MAE"]) if not baseline_table.empty else float("inf")
    baseline_spread = float(baseline_table.iloc[0]["mae_relative_spread"]) if not baseline_table.empty else float("inf")
    standards = evaluate_main_result_standards(final_best, baseline_best_mae, baseline_spread, thesis_contract)
    raw_standards = evaluate_main_result_standards(raw_best, baseline_best_mae, baseline_spread, thesis_contract) if raw_best else None
    paper_safe_best = None
    if not candidate_table.empty and "minimum_contract_pass" in candidate_table.columns:
        paper_safe_rows = candidate_table[candidate_table["minimum_contract_pass"] == True]  # noqa: E712
        if not paper_safe_rows.empty:
            paper_safe_best = paper_safe_rows.iloc[0].to_dict()

    summary: dict[str, Any] = {
        "metr_final_best": final_best,
        "metr_final_best_paper": final_best,
        "metr_final_best_raw": raw_best,
        "paper_safe_best": paper_safe_best,
        "paper_and_raw_same_candidate": bool(search_summary.get("paper_and_raw_same_candidate", False)),
        "strategy_leader": strategy_table.iloc[0].to_dict() if not strategy_table.empty else None,
        "window_mean_leader": window_table.iloc[0].to_dict() if not window_table.empty else None,
        "window_peak_leader": (
            window_table.sort_values(by="best_attack_success_rate", ascending=False).iloc[0].to_dict()
            if not window_table.empty
            else None
        ),
        **standards,
        "raw_contract_summary": raw_standards,
        "thesis_contract": thesis_contract,
        "candidate_count_in_paper_table": int(len(candidate_table)),
    }

    if cross_dir is not None and (cross_dir / "cross_dataset_summary.json").exists():
        cross_summary_path = cross_dir / "cross_dataset_summary.json"
        cross_summary = load_json(cross_summary_path)
        require_json_keys(cross_summary, ["final_best", "selected_candidate_count"], cross_summary_path)
        summary["cross_dataset_summary"] = cross_summary
        cross_best = dict(cross_summary.get("final_best", {}))
        summary.update(evaluate_cross_result_standards(cross_best, thesis_contract))
    return summary


def build_markdown_summary(
    baseline_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    strategy_table: pd.DataFrame,
    window_table: pd.DataFrame,
    defense_table: pd.DataFrame,
    cross_dir: Path | None,
    summary_payload: dict[str, Any],
) -> str:
    lines = ["# Thesis Experiment Summary", ""]
    if not baseline_table.empty:
        metr_row = baseline_table.iloc[0].to_dict()
        lines.append(
            f"- METR-LA frozen baseline: best MAE `{metr_row.get('best_MAE', 0):.4f}`, spread `{metr_row.get('mae_relative_spread', 0):.2%}`."
        )
    raw_best = summary_payload.get("metr_final_best_raw") or {}
    paper_best = summary_payload.get("metr_final_best_paper") or {}
    if raw_best:
        lines.append(
            f"- Raw champion: `{raw_best.get('sample_selection_mode')}` + `{raw_best.get('target_weight_mode')}` + `{raw_best.get('loss_focus_mode')}` gives legacy mean ASR `{float(raw_best.get('attack_success_rate', 0.0)):.2%}`, local raw-space ASR `{float(raw_best.get('raw_selected_nodes_tail_horizon_attack_success_rate', 0.0)):.2%}`, clean MAE drift `{float(raw_best.get('clean_MAE_delta_ratio', 0.0)):.2%}`, and local target attainment `{float(raw_best.get('raw_selected_nodes_tail_horizon_target_shift_attainment', 0.0)):.4f}`."
        )
    if paper_best:
        lines.append(
            f"- Paper champion: `{paper_best.get('sample_selection_mode')}` + `{paper_best.get('target_weight_mode')}` + `{paper_best.get('loss_focus_mode')}` gives legacy mean ASR `{float(paper_best.get('attack_success_rate', 0.0)):.2%}`, local raw-space ASR `{float(paper_best.get('raw_selected_nodes_tail_horizon_attack_success_rate', 0.0)):.2%}`, clean MAE drift `{float(paper_best.get('clean_MAE_delta_ratio', 0.0)):.2%}`, and local target attainment `{float(paper_best.get('raw_selected_nodes_tail_horizon_target_shift_attainment', 0.0)):.4f}`."
        )
    if summary_payload.get("paper_and_raw_same_candidate"):
        lines.append("- Raw champion and paper champion are the same candidate.")
    if not strategy_table.empty:
        lines.append(
            f"- Strategy comparison: `{strategy_table.iloc[0]['selection_strategy']}` has the highest mean local ASR `{strategy_table.iloc[0]['mean_local_attack_success_rate']:.2%}`."
        )
    if not window_table.empty:
        peak_leader = window_table.sort_values(by='best_local_attack_success_rate', ascending=False).iloc[0]
        lines.append(
            f"- Window-family tradeoff: mean local ASR leader is `{window_table.iloc[0]['window_mode']}` at `{window_table.iloc[0]['mean_local_attack_success_rate']:.2%}`, while peak local ASR leader is `{peak_leader['window_mode']}` at `{peak_leader['best_local_attack_success_rate']:.2%}`."
        )
    if not defense_table.empty:
        ma_row = defense_table[defense_table["name"] == "moving_average_asr_effect"]
        if not ma_row.empty:
            lines.append(
                f"- Simple smoothing ASR effect: `{float(ma_row.iloc[0]['asr_before']):.2%} -> {float(ma_row.iloc[0]['asr_after']):.2%}`."
            )
    if cross_dir is not None and "cross_dataset_summary" in summary_payload:
        cross_summary = summary_payload["cross_dataset_summary"]
        lines.append(
            f"- Cross-dataset replay: `{cross_summary.get('selected_candidate_count', 0)}` candidates replayed on the secondary dataset."
        )
        cross_best = cross_summary.get("final_best", {})
        lines.append(
            f"- Cross-dataset best local raw-space ASR: `{float(cross_best.get('raw_selected_nodes_tail_horizon_attack_success_rate', 0.0)):.2%}` with clean MAE drift `{float(cross_best.get('clean_MAE_delta_ratio', 0.0)):.2%}`."
        )
    lines.append(
        f"- Minimum paper bar: `{'met' if summary_payload.get('minimum_bar_met') else 'not met'}`; strong paper bar: `{'met' if summary_payload.get('strong_bar_met') else 'not met'}`."
    )
    lines.append(
        f"- Previous mainline local-ASR bar (`{float(summary_payload.get('thesis_contract', {}).get('previous_mainline_local_asr', 0.0)):.2%}`): `{'beaten' if summary_payload.get('beat_previous_local_mainline') else 'not beaten'}`."
    )
    lines.append(
        f"- Stop rule: `{'triggered' if summary_payload.get('stop_rule_triggered') else 'not triggered'}`; extra follow-up search recommendation: `{'yes' if summary_payload.get('follow_up_search_recommended') else 'no'}`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thesis_contract = resolve_thesis_contract()
    metr_baseline_dir = Path(args.metr_baseline_dir).resolve()
    metr_poison_dir = Path(args.metr_poison_dir).resolve()
    defense_dir = Path(args.defense_dir).resolve()
    cross_dir = Path(args.cross_dir).resolve() if args.cross_dir else None

    output_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(ROOT / "results", "thesis_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_table = build_baseline_table(metr_baseline_dir, cross_dir)
    search_summary_path = metr_poison_dir / "search_summary.json"
    if search_summary_path.exists():
        thesis_contract = resolve_thesis_contract(load_json(search_summary_path).get("thesis_contract", thesis_contract))
    candidate_table = build_candidate_table(metr_poison_dir, thesis_contract)
    family_table = build_family_comparison(metr_poison_dir, thesis_contract)
    strategy_table = build_single_axis_comparison(metr_poison_dir, "selection_strategy", thesis_contract)
    window_table = build_single_axis_comparison(metr_poison_dir, "window_mode", thesis_contract)
    target_horizon_table = build_single_axis_comparison(metr_poison_dir, "target_horizon_mode", thesis_contract)
    smoothing_table = build_single_axis_comparison(metr_poison_dir, "frequency_smoothing_strength", thesis_contract)
    parameter_table = build_parameter_sensitivity(metr_poison_dir, thesis_contract)
    defense_table = build_defense_table(defense_dir)

    save_table(baseline_table.to_dict(orient="records"), output_dir / "baseline_stability_table.csv")
    save_table(candidate_table.to_dict(orient="records"), output_dir / "paper_candidate_table.csv")
    save_table(family_table.to_dict(orient="records"), output_dir / "attack_family_comparison.csv")
    save_table(strategy_table.to_dict(orient="records"), output_dir / "selection_strategy_comparison.csv")
    save_table(window_table.to_dict(orient="records"), output_dir / "window_mode_comparison.csv")
    save_table(target_horizon_table.to_dict(orient="records"), output_dir / "target_horizon_mode_comparison.csv")
    save_table(smoothing_table.to_dict(orient="records"), output_dir / "frequency_smoothing_comparison.csv")
    save_table(parameter_table.to_dict(orient="records"), output_dir / "parameter_sensitivity_table.csv")
    save_table(defense_table.to_dict(orient="records"), output_dir / "defense_summary_table.csv")

    if cross_dir is not None:
        if (cross_dir / "cross_candidate_summary.csv").exists():
            cross_candidate_table = load_csv(cross_dir / "cross_candidate_summary.csv")
            save_table(cross_candidate_table.to_dict(orient="records"), output_dir / "cross_candidate_comparison.csv")
        if (cross_dir / "cross_family_summary.csv").exists():
            cross_family_table = load_csv(cross_dir / "cross_family_summary.csv")
            save_table(cross_family_table.to_dict(orient="records"), output_dir / "cross_family_summary.csv")

    summary_payload = build_summary_payload(
        baseline_table,
        metr_poison_dir,
        cross_dir,
        candidate_table,
        strategy_table,
        window_table,
        thesis_contract,
    )
    save_json(summary_payload, output_dir / "thesis_summary.json")
    (output_dir / "thesis_summary.md").write_text(
        build_markdown_summary(baseline_table, candidate_table, strategy_table, window_table, defense_table, cross_dir, summary_payload),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
