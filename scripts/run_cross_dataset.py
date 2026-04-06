from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.config import load_config
from traffic_poison.experiment import (
    build_model,
    dataset_summary,
    evaluate_on_arrays,
    make_loader,
    model_kwargs_from_bundle,
    prepare_bundle_from_config,
    relative_metric_change,
)
from traffic_poison.poisoning import (
    build_poisoned_training_set,
    compute_attack_evaluation_views,
    compute_attack_success_metrics,
    compute_prediction_shift_metrics,
    compute_stealth_metrics,
    rank_vulnerable_positions,
)
from traffic_poison.reporting import plot_prediction_case, save_table
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay selected attack families on a second dataset.")
    parser.add_argument("--config", required=True, help="Path to the secondary dataset config, usually PEMS-BAY.")
    parser.add_argument(
        "--best-attack-json",
        help="Optional path to best_attack.json for legacy single-candidate replay mode.",
    )
    parser.add_argument(
        "--source-poison-dir",
        help="Optional source poisoning directory. When set, the script selects multiple candidates from attack_results.csv.",
    )
    return parser.parse_args()


def score_row(row: dict[str, Any]) -> tuple[float, ...]:
    primary_local_asr = float(
        row.get(
            "raw_selected_nodes_tail_horizon_attack_success_rate",
            row.get(
                "raw_selected_nodes_attack_success_rate",
                row.get("raw_global_attack_success_rate", row.get("attack_success_rate", 0.0)),
            ),
        )
    )
    secondary_local_asr = float(
        row.get(
            "raw_selected_nodes_attack_success_rate",
            row.get("raw_global_attack_success_rate", row.get("attack_success_rate", 0.0)),
        )
    )
    primary_shift = float(
        row.get(
            "raw_selected_nodes_tail_horizon_target_shift_attainment",
            row.get(
                "raw_selected_nodes_target_shift_attainment",
                row.get("raw_global_target_shift_attainment", row.get("target_shift_attainment", 0.0)),
            ),
        )
    )
    return (
        primary_local_asr,
        secondary_local_asr,
        float(row.get("attack_success_rate", 0.0)),
        primary_shift,
        -abs(float(row.get("clean_MAE_delta_ratio", 0.0))),
        -float(row.get("frequency_energy_shift", row.get("anomaly_rate", 0.0))),
        -float(row.get("mean_z_score", 0.0)),
        -float(row.get("anomaly_rate", 0.0)),
    )


def within_clean_budget(row: dict[str, Any] | None, max_clean_mae_delta_ratio: float) -> bool:
    if row is None:
        return False
    return float(row.get("clean_MAE_delta_ratio", float("inf"))) <= max_clean_mae_delta_ratio


def row_sort_key(row: dict[str, Any], max_clean_mae_delta_ratio: float) -> tuple[float, ...]:
    return (
        1.0 if within_clean_budget(row, max_clean_mae_delta_ratio) else 0.0,
        *score_row(row),
    )


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "selection_strategy": str(row["selection_strategy"]),
        "poison_ratio": float(row["poison_ratio"]),
        "sigma_multiplier": float(row["sigma_multiplier"]),
        "target_shift_ratio": float(row["target_shift_ratio"]),
        "window_mode": str(row.get("window_mode", "tail")),
        "trigger_steps": int(float(row["trigger_steps"])),
        "trigger_node_count": int(float(row.get("trigger_node_count", 5))),
        "sample_selection_mode": str(row.get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(row.get("target_weight_mode", "flat")),
        "target_horizon_mode": str(row.get("target_horizon_mode", "all")),
        "target_horizon_count": int(float(row.get("target_horizon_count", 3))),
        "time_smoothing_kernel": int(float(row.get("time_smoothing_kernel", 3))),
        "frequency_smoothing_strength": float(row.get("frequency_smoothing_strength", 0.0)),
        "frequency_cutoff_ratio": float(row.get("frequency_cutoff_ratio", 0.5)),
        "frequency_decay": float(row.get("frequency_decay", 0.35)),
    }
    for key in ("local_forecast_error_mean", "global_forecast_error_mean", "selected_poison_score_mean"):
        if key in row and row[key] not in {None, ""}:
            candidate[key] = float(row[key])
    return candidate


def candidate_key(candidate: dict[str, Any]) -> str:
    serializable = {
        "selection_strategy": str(candidate["selection_strategy"]),
        "poison_ratio": round(float(candidate["poison_ratio"]), 8),
        "sigma_multiplier": round(float(candidate["sigma_multiplier"]), 8),
        "target_shift_ratio": round(float(candidate["target_shift_ratio"]), 8),
        "window_mode": str(candidate["window_mode"]),
        "trigger_steps": int(candidate["trigger_steps"]),
        "trigger_node_count": int(candidate["trigger_node_count"]),
        "sample_selection_mode": str(candidate.get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(candidate.get("target_weight_mode", "flat")),
        "target_horizon_mode": str(candidate["target_horizon_mode"]),
        "target_horizon_count": int(candidate["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate["time_smoothing_kernel"]),
        "frequency_smoothing_strength": round(float(candidate["frequency_smoothing_strength"]), 8),
        "frequency_cutoff_ratio": round(float(candidate["frequency_cutoff_ratio"]), 8),
        "frequency_decay": round(float(candidate.get("frequency_decay", 0.35)), 8),
    }
    return json.dumps(serializable, sort_keys=True)


def choose_best_row(rows: list[dict[str, Any]], max_clean_mae_delta_ratio: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row_sort_key(row, max_clean_mae_delta_ratio))


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for key in (
                "poison_ratio",
                "sigma_multiplier",
                "target_shift_ratio",
                "trigger_steps",
                "trigger_node_count",
                "sample_selection_mode",
                "target_weight_mode",
                "target_horizon_count",
                "time_smoothing_kernel",
                "attack_success_rate",
                "clean_MAE_delta_ratio",
                "frequency_energy_shift",
                "mean_z_score",
                "anomaly_rate",
                "frequency_smoothing_strength",
                "frequency_cutoff_ratio",
                "frequency_decay",
                "local_forecast_error_mean",
                "global_forecast_error_mean",
                "selected_poison_score_mean",
            ):
                if key in parsed and parsed[key] not in {None, ""}:
                    if key in {
                        "sample_selection_mode",
                        "target_weight_mode",
                    }:
                        parsed[key] = str(parsed[key])
                    else:
                        parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def select_candidates_from_source(source_poison_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    cross_cfg = dict(config.get("cross_validation", {}))
    source_table = str(cross_cfg.get("source_table", "attack_results.csv"))
    source_rows = load_source_rows(source_poison_dir / source_table)
    max_clean_mae_delta_ratio = float(cross_cfg.get("max_clean_mae_delta_ratio", 0.05))
    allowed_selection = {str(value) for value in cross_cfg.get("selection_strategies", ["error"])}
    allowed_poison_ratios = {round(float(value), 8) for value in cross_cfg.get("poison_ratios", [])}
    allowed_sigmas = {round(float(value), 8) for value in cross_cfg.get("sigma_multipliers", [])}
    allowed_node_counts = {int(value) for value in cross_cfg.get("trigger_node_count", [])}

    deduped: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if allowed_selection and str(row.get("selection_strategy")) not in allowed_selection:
            continue
        if allowed_poison_ratios and round(float(row.get("poison_ratio", 0.0)), 8) not in allowed_poison_ratios:
            continue
        if allowed_sigmas and round(float(row.get("sigma_multiplier", 0.0)), 8) not in allowed_sigmas:
            continue
        if allowed_node_counts and int(float(row.get("trigger_node_count", 0))) not in allowed_node_counts:
            continue
        if not within_clean_budget(row, max_clean_mae_delta_ratio):
            continue
        key = candidate_key(candidate_from_row(row))
        best_existing = deduped.get(key)
        if best_existing is None or row_sort_key(row, max_clean_mae_delta_ratio) > row_sort_key(best_existing, max_clean_mae_delta_ratio):
            deduped[key] = row

    family_order = [str(value) for value in cross_cfg.get("family_order", ["tail", "hybrid"])]
    family_top_k = int(cross_cfg.get("family_top_k", 2))
    family_rules = dict(cross_cfg.get("families", {}))

    selected_candidates: list[dict[str, Any]] = []
    for family_name in family_order:
        rules = dict(family_rules.get(family_name, {}))
        window_modes = {str(value) for value in rules.get("window_modes", [family_name])}
        trigger_steps = {int(value) for value in rules.get("trigger_steps", [])}

        family_rows = [
            row
            for row in deduped.values()
            if str(row.get("window_mode")) in window_modes
            and (not trigger_steps or int(float(row.get("trigger_steps", 0))) in trigger_steps)
        ]
        family_rows.sort(key=lambda row: row_sort_key(row, max_clean_mae_delta_ratio), reverse=True)

        for rank, row in enumerate(family_rows[:family_top_k], start=1):
            candidate = candidate_from_row(row)
            candidate["source_family"] = family_name
            candidate["source_family_rank"] = rank
            candidate["source_attack_success_rate"] = float(row.get("attack_success_rate", 0.0))
            candidate["source_clean_MAE_delta_ratio"] = float(row.get("clean_MAE_delta_ratio", 0.0))
            selected_candidates.append(candidate)
    return selected_candidates


def summarize_baseline_rows(rows: list[dict[str, Any]], dataset_name: str, best_seed: int, best_metrics: dict[str, float]) -> dict[str, Any]:
    maes = [float(row["MAE"]) for row in rows]
    mean_mae = float(np.mean(maes))
    relative_spread = float((max(maes) - min(maes)) / mean_mae) if mean_mae > 0 else 0.0
    return {
        "dataset_name": dataset_name,
        "repeat_count": int(len(rows)),
        "best_seed": int(best_seed),
        "best_MAE": float(best_metrics["MAE"]),
        "best_MAPE": float(best_metrics["MAPE"]),
        "best_RMSE": float(best_metrics["RMSE"]),
        "mae_mean": mean_mae,
        "mae_min": float(min(maes)),
        "mae_max": float(max(maes)),
        "mae_relative_spread": relative_spread,
        "stable_under_5_percent": relative_spread <= 0.05,
    }


def run_frozen_baseline(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    repeats = int(config["training"].get("repeats", 3))
    base_seed = int(config.get("seed", 42))
    rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None

    for repeat_idx in range(repeats):
        seed = base_seed + repeat_idx
        set_seed(seed)
        bundle = prepare_bundle_from_config(config, seed=seed)
        model = build_model(config, bundle)
        model_kwargs = model_kwargs_from_bundle(config, bundle)
        artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])
        clean_metrics, clean_true, clean_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
        _, _, clean_train_pred = evaluate_on_arrays(model, bundle.train_inputs, bundle.train_targets, config, artifacts.device)

        row = {
            "repeat": int(repeat_idx),
            "seed": int(seed),
            **clean_metrics,
            "best_epoch": int(artifacts.history.best_epoch),
            "best_val_loss": float(artifacts.history.best_val_loss),
        }
        rows.append(row)

        if best_payload is None or float(clean_metrics["MAE"]) < float(best_payload["clean_metrics"]["MAE"]):
            best_payload = {
                "seed": seed,
                "bundle": bundle,
                "model": model,
                "model_kwargs": model_kwargs,
                "artifacts": artifacts,
                "clean_metrics": clean_metrics,
                "clean_true": clean_true,
                "clean_pred": clean_pred,
                "clean_train_pred": clean_train_pred,
            }

    if best_payload is None:
        raise RuntimeError("No frozen baseline payload was produced.")

    summary = summarize_baseline_rows(
        rows,
        str(config["dataset"].get("name", "unknown")),
        int(best_payload["seed"]),
        best_payload["clean_metrics"],
    )
    return rows, summary, best_payload


def evaluate_attack_candidate(
    *,
    config: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_metrics: dict[str, float],
    candidate: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = baseline_payload["bundle"]
    target_horizon_candidates = int(config["poison"].get("target_horizon_candidates", max(int(candidate.get("target_horizon_count", 3)), 5)))
    target_horizon_offset = int(config["poison"].get("target_horizon_offset", 0))
    set_seed(seed)

    ranking = rank_vulnerable_positions(
        bundle.train_inputs,
        bundle.train_targets,
        baseline_payload["clean_train_pred"],
        bundle.adjacency,
        str(candidate["selection_strategy"]),
        int(candidate["trigger_node_count"]),
        int(candidate["trigger_steps"]),
        target_horizon_count=int(candidate["target_horizon_count"]),
        target_horizon_mode=str(candidate["target_horizon_mode"]),
        target_horizon_candidates=target_horizon_candidates,
        target_horizon_offset=target_horizon_offset,
    )

    poisoned_train = build_poisoned_training_set(
        bundle.train_inputs,
        bundle.train_targets,
        ranking["ranked_nodes"],
        float(candidate["poison_ratio"]),
        float(candidate["sigma_multiplier"]),
        bundle.feature_std,
        int(candidate["trigger_steps"]),
        float(candidate["target_shift_ratio"]),
        float(config["poison"].get("fallback_shift_ratio", 0.05)),
        ranked_windows=ranking["ranked_windows"],
        window_mode=str(candidate["window_mode"]),
        sample_selection_mode=str(candidate.get("sample_selection_mode", "input_energy")),
        target_horizon_mode=str(candidate["target_horizon_mode"]),
        target_horizon_count=int(candidate["target_horizon_count"]),
        target_horizon_indices=ranking.get("target_horizon_indices"),
        clean_predictions=baseline_payload["clean_train_pred"],
        target_weight_mode=str(candidate.get("target_weight_mode", "flat")),
        node_rank_weights=config["poison"].get("node_rank_weights"),
        tail_horizon_weights=config["poison"].get("tail_horizon_weights"),
        selection_tail_horizon_count=int(config["poison"].get("evaluation_tail_horizon_count", 3)),
        time_smoothing_kernel=int(candidate["time_smoothing_kernel"]),
        frequency_smoothing_strength=float(candidate["frequency_smoothing_strength"]),
        frequency_cutoff_ratio=float(candidate["frequency_cutoff_ratio"]),
        frequency_decay=float(candidate.get("frequency_decay", 0.35)),
    )
    poisoned_loader = make_loader(
        np.asarray(poisoned_train["poisoned_inputs"]),
        np.asarray(poisoned_train["poisoned_targets"]),
        batch_size=int(config["dataset"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config["dataset"].get("num_workers", 0)),
    )

    model = build_model(config, bundle)
    artifacts = train_model(model, poisoned_loader, bundle.val_loader, config["training"])
    poisoned_clean_metrics, _, poisoned_clean_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)

    triggered_test = build_poisoned_training_set(
        bundle.test_inputs,
        bundle.test_targets,
        ranking["ranked_nodes"],
        1.0,
        float(candidate["sigma_multiplier"]),
        bundle.feature_std,
        int(candidate["trigger_steps"]),
        float(candidate["target_shift_ratio"]),
        float(config["poison"].get("fallback_shift_ratio", 0.05)),
        ranked_windows=ranking["ranked_windows"],
        window_mode=str(candidate["window_mode"]),
        sample_selection_mode=str(candidate.get("sample_selection_mode", "input_energy")),
        target_horizon_mode=str(candidate["target_horizon_mode"]),
        target_horizon_count=int(candidate["target_horizon_count"]),
        target_horizon_indices=ranking.get("target_horizon_indices"),
        target_weight_mode=str(candidate.get("target_weight_mode", "flat")),
        node_rank_weights=config["poison"].get("node_rank_weights"),
        tail_horizon_weights=config["poison"].get("tail_horizon_weights"),
        selection_tail_horizon_count=int(config["poison"].get("evaluation_tail_horizon_count", 3)),
        time_smoothing_kernel=int(candidate["time_smoothing_kernel"]),
        frequency_smoothing_strength=float(candidate["frequency_smoothing_strength"]),
        frequency_cutoff_ratio=float(candidate["frequency_cutoff_ratio"]),
        frequency_decay=float(candidate.get("frequency_decay", 0.35)),
    )
    triggered_inputs = np.asarray(triggered_test["poisoned_inputs"])
    triggered_metrics, _, triggered_pred = evaluate_on_arrays(
        model,
        triggered_inputs,
        bundle.test_targets,
        config,
        artifacts.device,
    )

    asr = compute_attack_success_metrics(
        bundle.test_targets,
        triggered_pred,
        float(candidate["target_shift_ratio"]),
        float(config["poison"].get("success_tolerance_ratio", 0.03)),
    )
    shift_metrics = compute_prediction_shift_metrics(poisoned_clean_pred, triggered_pred, float(candidate["target_shift_ratio"]))
    eval_views = compute_attack_evaluation_views(
        bundle.test_targets,
        poisoned_clean_pred,
        triggered_pred,
        float(candidate["target_shift_ratio"]),
        float(config["poison"].get("success_tolerance_ratio", 0.03)),
        selected_nodes=poisoned_train["selected_nodes"],
        target_horizon_indices=poisoned_train["selected_target_horizon_indices"],
        tail_horizon_count=int(config["poison"].get("evaluation_tail_horizon_count", 3)),
        scaler=bundle.scaler,
    )
    stealth = compute_stealth_metrics(bundle.test_inputs, triggered_inputs)
    clean_delta = relative_metric_change(baseline_metrics, poisoned_clean_metrics)

    row = {
        "seed": int(seed),
        "selection_strategy": str(candidate["selection_strategy"]),
        "window_mode": str(candidate["window_mode"]),
        "trigger_steps": int(candidate["trigger_steps"]),
        "trigger_node_count": int(candidate["trigger_node_count"]),
        "sample_selection_mode": str(candidate.get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(candidate.get("target_weight_mode", "flat")),
        "target_horizon_mode": str(candidate["target_horizon_mode"]),
        "target_horizon_count": int(candidate["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate["time_smoothing_kernel"]),
        "frequency_smoothing_strength": float(candidate["frequency_smoothing_strength"]),
        "frequency_cutoff_ratio": float(candidate["frequency_cutoff_ratio"]),
        "frequency_decay": float(candidate.get("frequency_decay", 0.35)),
        "poison_ratio": float(candidate["poison_ratio"]),
        "sigma_multiplier": float(candidate["sigma_multiplier"]),
        "target_shift_ratio": float(candidate["target_shift_ratio"]),
        "source_family": str(candidate.get("source_family", "single_replay")),
        "source_family_rank": int(candidate.get("source_family_rank", 1)),
        "source_attack_success_rate": float(candidate.get("source_attack_success_rate", 0.0)),
        "source_clean_MAE_delta_ratio": float(candidate.get("source_clean_MAE_delta_ratio", 0.0)),
        "selected_nodes": ",".join(str(node) for node in poisoned_train["selected_nodes"]),
        "selected_time_indices": ",".join(str(idx) for idx in poisoned_train["selected_time_indices"]),
        "selected_target_horizon_indices": ",".join(str(idx) for idx in poisoned_train["selected_target_horizon_indices"]),
        "node_rank_weights": ",".join(f"{weight:.4f}" for weight in poisoned_train.get("node_rank_weights", [])),
        "tail_horizon_weights": ",".join(f"{weight:.4f}" for weight in poisoned_train.get("tail_horizon_weights", [])),
        **{f"baseline_{key}": float(value) for key, value in baseline_metrics.items()},
        **{f"clean_{key}": float(value) for key, value in poisoned_clean_metrics.items()},
        **{f"triggered_{key}": float(value) for key, value in triggered_metrics.items()},
        **{f"clean_{key}": float(value) for key, value in clean_delta.items()},
        "attack_success_rate": float(asr["attack_success_rate"]),
        "tolerance_ratio": float(asr["tolerance_ratio"]),
        "local_forecast_error_mean": float(poisoned_train.get("local_forecast_error_mean", 0.0)),
        "global_forecast_error_mean": float(poisoned_train.get("global_forecast_error_mean", 0.0)),
        "selected_poison_score_mean": float(poisoned_train.get("selected_poison_score_mean", 0.0)),
        **shift_metrics,
        **eval_views,
        **stealth,
    }

    payload = {
        "row": row,
        "triggered_pred": triggered_pred,
        "clean_pred": poisoned_clean_pred,
        "clean_true": baseline_payload["clean_true"],
    }
    return row, payload


def summarize_candidate_rows(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")

    summary = {
        "selection_strategy": str(candidate_rows[0]["selection_strategy"]),
        "window_mode": str(candidate_rows[0]["window_mode"]),
        "trigger_steps": int(candidate_rows[0]["trigger_steps"]),
        "trigger_node_count": int(candidate_rows[0]["trigger_node_count"]),
        "sample_selection_mode": str(candidate_rows[0].get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(candidate_rows[0].get("target_weight_mode", "flat")),
        "target_horizon_mode": str(candidate_rows[0]["target_horizon_mode"]),
        "target_horizon_count": int(candidate_rows[0]["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate_rows[0]["time_smoothing_kernel"]),
        "frequency_smoothing_strength": float(candidate_rows[0]["frequency_smoothing_strength"]),
        "frequency_cutoff_ratio": float(candidate_rows[0]["frequency_cutoff_ratio"]),
        "frequency_decay": float(candidate_rows[0].get("frequency_decay", 0.35)),
        "poison_ratio": float(candidate_rows[0]["poison_ratio"]),
        "sigma_multiplier": float(candidate_rows[0]["sigma_multiplier"]),
        "target_shift_ratio": float(candidate_rows[0]["target_shift_ratio"]),
        "source_family": str(candidate_rows[0]["source_family"]),
        "source_family_rank": int(candidate_rows[0]["source_family_rank"]),
        "repeat_count": int(len(candidate_rows)),
        "seed_list": ",".join(str(int(row["seed"])) for row in candidate_rows),
    }

    exclude = {
        "selection_strategy",
        "window_mode",
        "target_horizon_mode",
        "source_family",
        "source_family_rank",
        "repeat_count",
        "seed_list",
        "selected_nodes",
        "selected_time_indices",
        "selected_target_horizon_indices",
        "seed",
        "candidate_rank",
        "sample_selection_mode",
        "target_weight_mode",
    }
    numeric_keys = [
        key
        for key, value in candidate_rows[0].items()
        if key not in exclude and isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in candidate_rows]
        summary[key] = float(np.mean(values))
        if key in {
            "attack_success_rate",
            "clean_MAE_delta_ratio",
            "target_shift_attainment",
            "raw_selected_nodes_tail_horizon_attack_success_rate",
            "raw_selected_nodes_tail_horizon_target_shift_attainment",
            "raw_selected_nodes_attack_success_rate",
            "raw_selected_nodes_target_shift_attainment",
            "raw_global_attack_success_rate",
            "raw_global_target_shift_attainment",
        }:
            summary[f"{key}_min"] = float(min(values))
            summary[f"{key}_max"] = float(max(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def build_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family_rows.setdefault(str(row["source_family"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for family_name, items in family_rows.items():
        asr_values = [float(item["attack_success_rate"]) for item in items]
        local_asr_values = [float(item.get("raw_selected_nodes_tail_horizon_attack_success_rate", item["attack_success_rate"])) for item in items]
        clean_values = [float(item["clean_MAE_delta_ratio"]) for item in items]
        shift_values = [float(item.get("target_shift_attainment", 0.0)) for item in items]
        summaries.append(
            {
                "source_family": family_name,
                "candidate_count": len(items),
                "mean_attack_success_rate": float(np.mean(asr_values)),
                "best_attack_success_rate": float(max(asr_values)),
                "mean_local_attack_success_rate": float(np.mean(local_asr_values)),
                "best_local_attack_success_rate": float(max(local_asr_values)),
                "mean_clean_MAE_delta_ratio": float(np.mean(clean_values)),
                "mean_target_shift_attainment": float(np.mean(shift_values)),
            }
        )
    summaries.sort(key=lambda row: (row["mean_local_attack_success_rate"], row["mean_attack_success_rate"]), reverse=True)
    return summaries


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_cross")

    source_poison_dir = Path(args.source_poison_dir).resolve() if args.source_poison_dir else None
    if source_poison_dir is not None:
        selected_candidates = select_candidates_from_source(source_poison_dir, config)
        if not selected_candidates:
            raise RuntimeError("No cross-dataset candidates matched the configured family filters.")
    elif args.best_attack_json:
        with Path(args.best_attack_json).resolve().open("r", encoding="utf-8") as handle:
            best_attack = json.load(handle)
        candidate = candidate_from_row(best_attack)
        candidate["source_family"] = "single_replay"
        candidate["source_family_rank"] = 1
        candidate["source_attack_success_rate"] = float(best_attack.get("attack_success_rate", 0.0))
        candidate["source_clean_MAE_delta_ratio"] = float(best_attack.get("clean_MAE_delta_ratio", 0.0))
        selected_candidates = [candidate]
    else:
        raise ValueError("Either --source-poison-dir or --best-attack-json must be provided.")

    baseline_rows, baseline_summary, baseline_payload = run_frozen_baseline(config)
    baseline_metrics = baseline_payload["clean_metrics"]
    max_clean_mae_delta_ratio = float(config.get("cross_validation", {}).get("max_clean_mae_delta_ratio", 0.05))
    attack_repeats = int(config.get("cross_validation", {}).get("attack_repeats", config["training"].get("repeats", 3)))
    base_seed = int(config.get("seed", 42))

    save_table(selected_candidates, run_dir / "source_candidate_table.csv")
    save_table(baseline_rows, run_dir / "cross_baseline_repeats.csv")
    save_table([baseline_summary], run_dir / "cross_baseline_stability.csv")
    save_json(baseline_summary, run_dir / "cross_baseline_summary.json")

    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_summary: dict[str, Any] | None = None

    for candidate_rank, candidate in enumerate(selected_candidates, start=1):
        candidate_rows: list[dict[str, Any]] = []
        candidate_best_payload: dict[str, Any] | None = None
        candidate_best_row: dict[str, Any] | None = None

        for repeat_idx in range(attack_repeats):
            row, payload = evaluate_attack_candidate(
                config=config,
                baseline_payload=baseline_payload,
                baseline_metrics=baseline_metrics,
                candidate=candidate,
                seed=base_seed + repeat_idx,
            )
            row["candidate_rank"] = int(candidate_rank)
            row["repeat_index"] = int(repeat_idx)
            candidate_rows.append(row)
            repeat_rows.append(row)

            if candidate_best_row is None or row_sort_key(row, max_clean_mae_delta_ratio) > row_sort_key(candidate_best_row, max_clean_mae_delta_ratio):
                candidate_best_row = row
                candidate_best_payload = payload

        summary = summarize_candidate_rows(candidate_rows)
        summary["candidate_rank"] = int(candidate_rank)
        summary_rows.append(summary)

        if best_summary is None or row_sort_key(summary, max_clean_mae_delta_ratio) > row_sort_key(best_summary, max_clean_mae_delta_ratio):
            best_summary = summary
            best_payload = candidate_best_payload

    if best_summary is None or best_payload is None:
        raise RuntimeError("Cross-dataset replay did not produce any valid summary rows.")

    family_summary = build_family_summary(summary_rows)
    final_best = choose_best_row(summary_rows, max_clean_mae_delta_ratio)

    save_table(repeat_rows, run_dir / "cross_candidate_repeats.csv")
    save_table(summary_rows, run_dir / "cross_candidate_summary.csv")
    save_table(family_summary, run_dir / "cross_family_summary.csv")
    save_json(
        {
            "dataset_summary": dataset_summary(baseline_payload["bundle"], config),
            "baseline_summary": baseline_summary,
            "selected_candidate_count": len(selected_candidates),
            "family_summary": family_summary,
            "final_best": final_best,
            "source_poison_dir": str(source_poison_dir) if source_poison_dir else None,
        },
        run_dir / "cross_dataset_summary.json",
    )
    plot_prediction_case(
        baseline_payload["clean_true"],
        baseline_payload["clean_pred"],
        run_dir / "cross_dataset_prediction_case.png",
        poisoned_pred=best_payload["triggered_pred"],
        sample_index=0,
        node_index=0,
        title="Cross-Dataset Replay",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
