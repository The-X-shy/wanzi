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
from traffic_poison.thesis_contract import (
    candidate_contract_flags,
    choose_best_row,
    eligible_for_cross_replay,
    evaluate_cross_result_standards,
    resolve_thesis_contract,
    row_sort_key,
    within_clean_budget,
)
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
        "headroom_error_mix": float(row.get("headroom_error_mix", 0.6)),
        "global_shift_fraction": float(row.get("global_shift_fraction", 0.3)),
        "tail_focus_multiplier": float(row.get("tail_focus_multiplier", 1.6)),
        "loss_focus_mode": str(row.get("loss_focus_mode", "uniform")),
        "loss_selected_node_weight": float(row.get("loss_selected_node_weight", 1.15)),
        "loss_tail_horizon_weight": float(row.get("loss_tail_horizon_weight", 1.75)),
        "loss_headroom_boost": float(row.get("loss_headroom_boost", 0.4)),
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
        "headroom_error_mix": round(float(candidate.get("headroom_error_mix", 0.6)), 8),
        "global_shift_fraction": round(float(candidate.get("global_shift_fraction", 0.3)), 8),
        "tail_focus_multiplier": round(float(candidate.get("tail_focus_multiplier", 1.6)), 8),
        "loss_focus_mode": str(candidate.get("loss_focus_mode", "uniform")),
        "loss_selected_node_weight": round(float(candidate.get("loss_selected_node_weight", 1.15)), 8),
        "loss_tail_horizon_weight": round(float(candidate.get("loss_tail_horizon_weight", 1.75)), 8),
        "loss_headroom_boost": round(float(candidate.get("loss_headroom_boost", 0.4)), 8),
    }
    return json.dumps(serializable, sort_keys=True)


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
                "headroom_error_mix",
                "global_shift_fraction",
                "tail_focus_multiplier",
                "local_forecast_error_mean",
                "global_forecast_error_mean",
                "selected_poison_score_mean",
                "loss_selected_node_weight",
                "loss_tail_horizon_weight",
                "loss_headroom_boost",
            ):
                if key in parsed and parsed[key] not in {None, ""}:
                    if key in {
                        "sample_selection_mode",
                        "target_weight_mode",
                        "loss_focus_mode",
                    }:
                        parsed[key] = str(parsed[key])
                    else:
                        parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def select_candidates_from_source(
    source_poison_dir: Path,
    config: dict[str, Any],
    thesis_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    cross_cfg = dict(config.get("cross_validation", {}))
    source_table = str(cross_cfg.get("source_table", "attack_results.csv"))
    source_rows = load_source_rows(source_poison_dir / source_table)
    allowed_selection = {str(value) for value in cross_cfg.get("selection_strategies", ["error"])}
    allowed_poison_ratios = {round(float(value), 8) for value in cross_cfg.get("poison_ratios", [])}
    allowed_sigmas = {round(float(value), 8) for value in cross_cfg.get("sigma_multipliers", [])}
    allowed_node_counts = {int(value) for value in cross_cfg.get("trigger_node_count", [])}
    enforce_source_contract_gate = bool(cross_cfg.get("enforce_source_contract_gate", False))

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
        if not within_clean_budget(row, thesis_contract):
            continue
        if enforce_source_contract_gate and not eligible_for_cross_replay(row, thesis_contract):
            continue
        key = candidate_key(candidate_from_row(row))
        best_existing = deduped.get(key)
        if best_existing is None or row_sort_key(row, thesis_contract) > row_sort_key(best_existing, thesis_contract):
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
        family_rows.sort(key=lambda row: row_sort_key(row, thesis_contract), reverse=True)

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
    thesis_contract: dict[str, Any],
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
        headroom_floor=float(config["poison"].get("headroom_floor", 0.0)),
        headroom_error_mix=float(candidate.get("headroom_error_mix", config["poison"].get("headroom_error_mix", 0.6))),
        global_shift_fraction=float(candidate.get("global_shift_fraction", config["poison"].get("global_shift_fraction", 0.3))),
        tail_focus_multiplier=float(candidate.get("tail_focus_multiplier", config["poison"].get("tail_focus_multiplier", 1.6))),
        loss_focus_mode=str(candidate.get("loss_focus_mode", config["poison"].get("loss_focus_mode", "uniform"))),
        loss_selected_node_weight=float(candidate.get("loss_selected_node_weight", config["poison"].get("loss_selected_node_weight", 1.15))),
        loss_tail_horizon_weight=float(candidate.get("loss_tail_horizon_weight", config["poison"].get("loss_tail_horizon_weight", 1.75))),
        loss_headroom_boost=float(candidate.get("loss_headroom_boost", config["poison"].get("loss_headroom_boost", 0.4))),
        feature_scaler=bundle.scaler,
        trigger_feature_std=bundle.feature_std,
    )
    poisoned_loader = make_loader(
        np.asarray(poisoned_train["poisoned_inputs"]),
        np.asarray(poisoned_train["poisoned_targets"]),
        loss_weights=np.asarray(poisoned_train["poisoned_loss_weights"]),
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
        headroom_floor=float(config["poison"].get("headroom_floor", 0.0)),
        headroom_error_mix=float(candidate.get("headroom_error_mix", config["poison"].get("headroom_error_mix", 0.6))),
        global_shift_fraction=float(candidate.get("global_shift_fraction", config["poison"].get("global_shift_fraction", 0.3))),
        tail_focus_multiplier=float(candidate.get("tail_focus_multiplier", config["poison"].get("tail_focus_multiplier", 1.6))),
        loss_focus_mode=str(candidate.get("loss_focus_mode", config["poison"].get("loss_focus_mode", "uniform"))),
        loss_selected_node_weight=float(candidate.get("loss_selected_node_weight", config["poison"].get("loss_selected_node_weight", 1.15))),
        loss_tail_horizon_weight=float(candidate.get("loss_tail_horizon_weight", config["poison"].get("loss_tail_horizon_weight", 1.75))),
        loss_headroom_boost=float(candidate.get("loss_headroom_boost", config["poison"].get("loss_headroom_boost", 0.4))),
        feature_scaler=bundle.scaler,
        trigger_feature_std=bundle.feature_std,
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
        "headroom_error_mix": float(candidate.get("headroom_error_mix", 0.6)),
        "global_shift_fraction": float(candidate.get("global_shift_fraction", 0.3)),
        "tail_focus_multiplier": float(candidate.get("tail_focus_multiplier", 1.6)),
        "loss_focus_mode": str(candidate.get("loss_focus_mode", "uniform")),
        "loss_selected_node_weight": float(candidate.get("loss_selected_node_weight", 1.15)),
        "loss_tail_horizon_weight": float(candidate.get("loss_tail_horizon_weight", 1.75)),
        "loss_headroom_boost": float(candidate.get("loss_headroom_boost", 0.4)),
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
        "attack_success_rate": float(eval_views["raw_global_attack_success_rate"]),
        "scaled_attack_success_rate": float(asr["attack_success_rate"]),
        "elementwise_attack_success_rate": float(eval_views["raw_global_elementwise_attack_success_rate"]),
        "sample_all_attack_success_rate": float(eval_views["raw_global_sample_all_attack_success_rate"]),
        "tolerance_ratio": float(asr["tolerance_ratio"]),
        "local_forecast_error_mean": float(poisoned_train.get("local_forecast_error_mean", 0.0)),
        "global_forecast_error_mean": float(poisoned_train.get("global_forecast_error_mean", 0.0)),
        "selected_poison_score_mean": float(poisoned_train.get("selected_poison_score_mean", 0.0)),
        "positive_headroom_rate": float(poisoned_train.get("positive_headroom_rate", 0.0)),
        "selected_headroom_mean": float(poisoned_train.get("selected_headroom_mean", 0.0)),
        "selected_headroom_score_mean": float(poisoned_train.get("selected_headroom_score_mean", 0.0)),
        **shift_metrics,
        **eval_views,
        **stealth,
    }
    row.update(candidate_contract_flags(row, thesis_contract))

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
        "loss_focus_mode": str(candidate_rows[0].get("loss_focus_mode", "uniform")),
        "loss_selected_node_weight": float(candidate_rows[0].get("loss_selected_node_weight", 1.15)),
        "loss_tail_horizon_weight": float(candidate_rows[0].get("loss_tail_horizon_weight", 1.75)),
        "loss_headroom_boost": float(candidate_rows[0].get("loss_headroom_boost", 0.4)),
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
    thesis_contract = resolve_thesis_contract(config)
    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_cross")

    source_poison_dir = Path(args.source_poison_dir).resolve() if args.source_poison_dir else None
    if source_poison_dir is not None:
        selected_candidates = select_candidates_from_source(source_poison_dir, config, thesis_contract)
        if not selected_candidates:
            raise RuntimeError("No cross-dataset candidates passed the configured family filters and thesis replay gate.")
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
                thesis_contract=thesis_contract,
                baseline_payload=baseline_payload,
                baseline_metrics=baseline_metrics,
                candidate=candidate,
                seed=base_seed + repeat_idx,
            )
            row["candidate_rank"] = int(candidate_rank)
            row["repeat_index"] = int(repeat_idx)
            candidate_rows.append(row)
            repeat_rows.append(row)

            if candidate_best_row is None or row_sort_key(row, thesis_contract) > row_sort_key(candidate_best_row, thesis_contract):
                candidate_best_row = row
                candidate_best_payload = payload

        summary = summarize_candidate_rows(candidate_rows)
        summary.update(candidate_contract_flags(summary, thesis_contract))
        summary["candidate_rank"] = int(candidate_rank)
        summary_rows.append(summary)

        if best_summary is None or row_sort_key(summary, thesis_contract) > row_sort_key(best_summary, thesis_contract):
            best_summary = summary
            best_payload = candidate_best_payload

    if best_summary is None or best_payload is None:
        raise RuntimeError("Cross-dataset replay did not produce any valid summary rows.")

    family_summary = build_family_summary(summary_rows)
    final_best = choose_best_row(summary_rows, thesis_contract)
    cross_contract_summary = evaluate_cross_result_standards(final_best, thesis_contract)

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
            "thesis_contract": thesis_contract,
            "contract_summary": cross_contract_summary,
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
