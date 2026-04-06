from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.config import load_config
from traffic_poison.experiment import (
    build_model,
    build_model_from_kwargs,
    dataset_summary,
    evaluate_on_arrays,
    make_loader,
    model_kwargs_from_bundle,
    prepare_bundle_from_config,
    relative_metric_change,
    write_markdown_summary,
)
from traffic_poison.poisoning import (
    build_poisoned_training_set,
    compute_attack_evaluation_views,
    compute_attack_success_metrics,
    compute_prediction_shift_metrics,
    compute_stealth_metrics,
    rank_vulnerable_positions,
)
from traffic_poison.reporting import plot_prediction_case, plot_trigger_case, save_table
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, flatten_metrics, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run poisoning sweeps for the traffic forecasting baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--baseline-dir", help="Optional directory produced by run_clean_baseline.py.")
    return parser.parse_args()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


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


def choose_best_row(rows: list[dict[str, Any]], max_clean_mae_delta_ratio: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row_sort_key(row, max_clean_mae_delta_ratio))


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_strategy": str(row["selection_strategy"]),
        "poison_ratio": float(row["poison_ratio"]),
        "sigma_multiplier": float(row["sigma_multiplier"]),
        "target_shift_ratio": float(row["target_shift_ratio"]),
        "window_mode": str(row.get("window_mode", "tail")),
        "sample_selection_mode": str(row.get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(row.get("target_weight_mode", "flat")),
        "trigger_steps": int(row["trigger_steps"]),
        "trigger_node_count": int(row.get("trigger_node_count", 5)),
        "target_horizon_mode": str(row.get("target_horizon_mode", "all")),
        "target_horizon_count": int(row.get("target_horizon_count", 3)),
        "time_smoothing_kernel": int(row.get("time_smoothing_kernel", 3)),
        "frequency_smoothing_strength": float(row.get("frequency_smoothing_strength", 0.0)),
        "frequency_cutoff_ratio": float(row.get("frequency_cutoff_ratio", 0.5)),
        "frequency_decay": float(row.get("frequency_decay", 0.35)),
    }


def normalize_candidate(candidate: dict[str, Any], poison_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_strategy": str(candidate.get("selection_strategy", poison_cfg.get("selection_strategies", ["error"])[0])),
        "poison_ratio": float(candidate.get("poison_ratio", poison_cfg.get("poison_ratios", [0.02])[0])),
        "sigma_multiplier": float(candidate.get("sigma_multiplier", poison_cfg.get("sigma_multipliers", [0.06])[0])),
        "target_shift_ratio": float(
            candidate.get(
                "target_shift_ratio",
                poison_cfg.get("target_shift_ratios", [poison_cfg.get("target_shift_ratio", 0.10)])[0],
            )
        ),
        "window_mode": str(candidate.get("window_mode", poison_cfg.get("window_modes", ["tail"])[0])),
        "sample_selection_mode": str(
            candidate.get(
                "sample_selection_mode",
                poison_cfg.get("sample_selection_modes", [poison_cfg.get("sample_selection_mode", "input_energy")])[0],
            )
        ),
        "target_weight_mode": str(
            candidate.get(
                "target_weight_mode",
                poison_cfg.get("target_weight_modes", [poison_cfg.get("target_weight_mode", "flat")])[0],
            )
        ),
        "trigger_steps": int(candidate.get("trigger_steps", poison_cfg.get("trigger_steps", 3))),
        "trigger_node_count": int(candidate.get("trigger_node_count", poison_cfg.get("trigger_node_count", 5))),
        "target_horizon_mode": str(
            candidate.get("target_horizon_mode", poison_cfg.get("target_horizon_modes", [poison_cfg.get("target_horizon_mode", "all")])[0])
        ),
        "target_horizon_count": int(candidate.get("target_horizon_count", poison_cfg.get("target_horizon_count", 3))),
        "time_smoothing_kernel": int(
            candidate.get("time_smoothing_kernel", poison_cfg.get("time_smoothing_kernels", [poison_cfg.get("time_smoothing_kernel", 3)])[0])
        ),
        "frequency_smoothing_strength": float(
            candidate.get(
                "frequency_smoothing_strength",
                poison_cfg.get("frequency_smoothing_strengths", [poison_cfg.get("frequency_smoothing_strength", 0.0)])[0],
            )
        ),
        "frequency_cutoff_ratio": float(
            candidate.get(
                "frequency_cutoff_ratio",
                poison_cfg.get("frequency_cutoff_ratios", [poison_cfg.get("frequency_cutoff_ratio", 0.5)])[0],
            )
        ),
        "frequency_decay": float(
            candidate.get(
                "frequency_decay",
                poison_cfg.get("frequency_decays", [poison_cfg.get("frequency_decay", 0.35)])[0],
            )
        ),
    }


def candidate_key(candidate: dict[str, Any]) -> str:
    serializable = {
        "selection_strategy": str(candidate["selection_strategy"]),
        "poison_ratio": round(float(candidate["poison_ratio"]), 8),
        "sigma_multiplier": round(float(candidate["sigma_multiplier"]), 8),
        "target_shift_ratio": round(float(candidate["target_shift_ratio"]), 8),
        "window_mode": str(candidate["window_mode"]),
        "sample_selection_mode": str(candidate["sample_selection_mode"]),
        "target_weight_mode": str(candidate["target_weight_mode"]),
        "trigger_steps": int(candidate["trigger_steps"]),
        "trigger_node_count": int(candidate["trigger_node_count"]),
        "target_horizon_mode": str(candidate["target_horizon_mode"]),
        "target_horizon_count": int(candidate["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate["time_smoothing_kernel"]),
        "frequency_smoothing_strength": round(float(candidate["frequency_smoothing_strength"]), 8),
        "frequency_cutoff_ratio": round(float(candidate["frequency_cutoff_ratio"]), 8),
        "frequency_decay": round(float(candidate.get("frequency_decay", 0.35)), 8),
    }
    return json.dumps(serializable, sort_keys=True)


def stage_name_from_cfg(stage_cfg: dict[str, Any], index: int) -> str:
    return str(stage_cfg.get("name", f"stage_{index + 1}"))


def build_search_stages(poison_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configured_stages = poison_cfg.get("search_stages")
    if isinstance(configured_stages, list) and configured_stages:
        return [dict(stage) for stage in configured_stages]
    return [
        {
            "name": "single_stage",
            "selection_strategies": poison_cfg.get("selection_strategies", []),
            "poison_ratios": poison_cfg.get("poison_ratios", []),
            "sigma_multipliers": poison_cfg.get("sigma_multipliers", []),
            "target_shift_ratios": poison_cfg.get(
                "target_shift_ratios",
                [poison_cfg.get("target_shift_ratio", 0.10)],
            ),
            "window_modes": poison_cfg.get("window_modes", ["tail"]),
            "sample_selection_modes": poison_cfg.get(
                "sample_selection_modes",
                [poison_cfg.get("sample_selection_mode", "input_energy")],
            ),
            "target_weight_modes": poison_cfg.get(
                "target_weight_modes",
                [poison_cfg.get("target_weight_mode", "flat")],
            ),
            "trigger_steps": poison_cfg.get("trigger_steps", 3),
            "trigger_node_count": poison_cfg.get("trigger_node_count", 5),
            "target_horizon_modes": poison_cfg.get("target_horizon_modes", [poison_cfg.get("target_horizon_mode", "all")]),
            "target_horizon_count": poison_cfg.get("target_horizon_count", 3),
            "time_smoothing_kernels": poison_cfg.get("time_smoothing_kernels", [poison_cfg.get("time_smoothing_kernel", 3)]),
            "frequency_smoothing_strengths": poison_cfg.get(
                "frequency_smoothing_strengths",
                [poison_cfg.get("frequency_smoothing_strength", 0.0)],
            ),
            "frequency_cutoff_ratios": poison_cfg.get("frequency_cutoff_ratios", [poison_cfg.get("frequency_cutoff_ratio", 0.5)]),
            "frequency_decays": poison_cfg.get("frequency_decays", [poison_cfg.get("frequency_decay", 0.35)]),
            "refine_from_previous": False,
        }
    ]


def resolve_stage_candidates(
    stage_cfg: dict[str, Any],
    poison_cfg: dict[str, Any],
    previous_best_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    explicit_candidates = stage_cfg.get("explicit_candidates")
    if isinstance(explicit_candidates, list) and explicit_candidates:
        deduped: list[dict[str, Any]] = []
        seen_explicit: set[str] = set()
        for raw_candidate in explicit_candidates:
            if not isinstance(raw_candidate, dict):
                continue
            candidate = normalize_candidate(raw_candidate, poison_cfg)
            key = candidate_key(candidate)
            if key in seen_explicit:
                continue
            seen_explicit.add(key)
            deduped.append(candidate)
        return deduped

    candidate_groups = stage_cfg.get("candidate_groups")
    if isinstance(candidate_groups, list) and candidate_groups:
        merged_candidates: list[dict[str, Any]] = []
        seen_group_candidates: set[str] = set()
        for group_cfg in candidate_groups:
            if not isinstance(group_cfg, dict):
                continue
            combined_cfg = {
                **{key: value for key, value in stage_cfg.items() if key not in {"candidate_groups", "explicit_candidates"}},
                **group_cfg,
            }
            group_candidates = resolve_stage_candidates(combined_cfg, poison_cfg, previous_best_row)
            for candidate in group_candidates:
                key = candidate_key(candidate)
                if key in seen_group_candidates:
                    continue
                seen_group_candidates.add(key)
                merged_candidates.append(candidate)
        return merged_candidates

    refine_from_previous = bool(stage_cfg.get("refine_from_previous", False)) and previous_best_row is not None

    def resolve_values(stage_key: str, default_value: Any, previous_key: str | None = None) -> list[Any]:
        if stage_key in stage_cfg:
            values = as_list(stage_cfg[stage_key])
        elif refine_from_previous and previous_key is not None:
            values = [previous_best_row[previous_key]]  # type: ignore[index]
        else:
            values = as_list(default_value)
        return values

    selection_strategies = [
        str(value)
        for value in resolve_values("selection_strategies", poison_cfg.get("selection_strategies", []), "selection_strategy")
    ]
    poison_ratios = [
        float(value)
        for value in resolve_values("poison_ratios", poison_cfg.get("poison_ratios", []), "poison_ratio")
    ]
    sigma_multipliers = [
        float(value)
        for value in resolve_values("sigma_multipliers", poison_cfg.get("sigma_multipliers", []), "sigma_multiplier")
    ]
    target_shift_ratios = [
        float(value)
        for value in resolve_values(
            "target_shift_ratios",
            poison_cfg.get("target_shift_ratios", [poison_cfg.get("target_shift_ratio", 0.10)]),
            "target_shift_ratio",
        )
    ]
    window_modes = [
        str(value)
        for value in resolve_values("window_modes", poison_cfg.get("window_modes", ["tail"]), "window_mode")
    ]
    sample_selection_modes = [
        str(value)
        for value in resolve_values(
            "sample_selection_modes",
            poison_cfg.get("sample_selection_modes", [poison_cfg.get("sample_selection_mode", "input_energy")]),
            "sample_selection_mode",
        )
    ]
    target_weight_modes = [
        str(value)
        for value in resolve_values(
            "target_weight_modes",
            poison_cfg.get("target_weight_modes", [poison_cfg.get("target_weight_mode", "flat")]),
            "target_weight_mode",
        )
    ]
    trigger_steps = [
        int(value)
        for value in resolve_values("trigger_steps", poison_cfg.get("trigger_steps", 3), "trigger_steps")
    ]
    trigger_node_count = [
        int(value)
        for value in resolve_values("trigger_node_count", poison_cfg.get("trigger_node_count", 5), "trigger_node_count")
    ]
    target_horizon_modes = [
        str(value)
        for value in resolve_values(
            "target_horizon_modes",
            poison_cfg.get("target_horizon_modes", [poison_cfg.get("target_horizon_mode", "all")]),
            "target_horizon_mode",
        )
    ]
    target_horizon_count = [
        int(value)
        for value in resolve_values(
            "target_horizon_count",
            poison_cfg.get("target_horizon_count", 3),
            "target_horizon_count",
        )
    ]
    time_smoothing_kernels = [
        int(value)
        for value in resolve_values(
            "time_smoothing_kernels",
            poison_cfg.get("time_smoothing_kernels", [poison_cfg.get("time_smoothing_kernel", 3)]),
            "time_smoothing_kernel",
        )
    ]
    frequency_smoothing_strengths = [
        float(value)
        for value in resolve_values(
            "frequency_smoothing_strengths",
            poison_cfg.get("frequency_smoothing_strengths", [poison_cfg.get("frequency_smoothing_strength", 0.0)]),
            "frequency_smoothing_strength",
        )
    ]
    frequency_cutoff_ratios = [
        float(value)
        for value in resolve_values(
            "frequency_cutoff_ratios",
            poison_cfg.get("frequency_cutoff_ratios", [poison_cfg.get("frequency_cutoff_ratio", 0.5)]),
            "frequency_cutoff_ratio",
        )
    ]
    frequency_decays = [
        float(value)
        for value in resolve_values(
            "frequency_decays",
            poison_cfg.get("frequency_decays", [poison_cfg.get("frequency_decay", 0.35)]),
            "frequency_decay",
        )
    ]

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in itertools.product(
        selection_strategies,
        poison_ratios,
        sigma_multipliers,
        target_shift_ratios,
        window_modes,
        sample_selection_modes,
        target_weight_modes,
        trigger_steps,
        trigger_node_count,
        target_horizon_modes,
        target_horizon_count,
        time_smoothing_kernels,
        frequency_smoothing_strengths,
        frequency_cutoff_ratios,
        frequency_decays,
    ):
        candidate = {
            "selection_strategy": combo[0],
            "poison_ratio": float(combo[1]),
            "sigma_multiplier": float(combo[2]),
            "target_shift_ratio": float(combo[3]),
            "window_mode": combo[4],
            "sample_selection_mode": combo[5],
            "target_weight_mode": combo[6],
            "trigger_steps": int(combo[7]),
            "trigger_node_count": int(combo[8]),
            "target_horizon_mode": combo[9],
            "target_horizon_count": int(combo[10]),
            "time_smoothing_kernel": int(combo[11]),
            "frequency_smoothing_strength": float(combo[12]),
            "frequency_cutoff_ratio": float(combo[13]),
            "frequency_decay": float(combo[14]),
        }
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def load_or_train_baseline(config: dict[str, Any], baseline_dir: str | None):
    bundle = prepare_bundle_from_config(config, seed=int(config.get("seed", 42)))

    if baseline_dir:
        baseline_path = Path(baseline_dir).resolve()
        checkpoint = torch.load(baseline_path / "clean_model.pt", map_location="cpu")
        model = build_model_from_kwargs(checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])

        train_predictions = np.load(baseline_path / "train_predictions.npz")
        clean_train_pred = train_predictions["y_pred"]
        baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device="cpu")
        return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, baseline_path

    set_seed(int(config.get("seed", 42)))
    model = build_model(config, bundle)
    artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])
    baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
    _, _, clean_train_pred = evaluate_on_arrays(model, bundle.train_inputs, bundle.train_targets, config, artifacts.device)
    return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, None


def evaluate_attack_candidate(
    *,
    config: dict[str, Any],
    bundle,
    baseline_metrics: dict[str, float],
    baseline_true: np.ndarray,
    baseline_pred: np.ndarray,
    clean_train_pred: np.ndarray,
    candidate: dict[str, Any],
    stage_name: str,
    seed: int,
    capture_payload: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    poison_cfg = config["poison"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    target_horizon_candidates = int(poison_cfg.get("target_horizon_candidates", max(int(candidate.get("target_horizon_count", 3)), 5)))
    target_horizon_offset = int(poison_cfg.get("target_horizon_offset", 0))

    set_seed(seed)
    ranking = rank_vulnerable_positions(
        bundle.train_inputs,
        bundle.train_targets,
        clean_train_pred,
        bundle.adjacency,
        candidate["selection_strategy"],
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
        float(poison_cfg.get("fallback_shift_ratio", 0.05)),
        ranked_windows=ranking["ranked_windows"],
        window_mode=str(candidate["window_mode"]),
        sample_selection_mode=str(candidate["sample_selection_mode"]),
        target_horizon_mode=str(candidate["target_horizon_mode"]),
        target_horizon_count=int(candidate["target_horizon_count"]),
        target_horizon_indices=ranking.get("target_horizon_indices"),
        clean_predictions=clean_train_pred,
        target_weight_mode=str(candidate["target_weight_mode"]),
        node_rank_weights=poison_cfg.get("node_rank_weights"),
        tail_horizon_weights=poison_cfg.get("tail_horizon_weights"),
        selection_tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
        time_smoothing_kernel=int(candidate["time_smoothing_kernel"]),
        frequency_smoothing_strength=float(candidate["frequency_smoothing_strength"]),
        frequency_cutoff_ratio=float(candidate["frequency_cutoff_ratio"]),
        frequency_decay=float(candidate.get("frequency_decay", 0.35)),
    )

    poisoned_loader = make_loader(
        np.asarray(poisoned_train["poisoned_inputs"]),
        np.asarray(poisoned_train["poisoned_targets"]),
        batch_size=int(dataset_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
    )

    model = build_model(config, bundle)
    artifacts = train_model(model, poisoned_loader, bundle.val_loader, training_cfg)
    clean_metrics, clean_true, clean_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)

    triggered_test = build_poisoned_training_set(
        bundle.test_inputs,
        bundle.test_targets,
        ranking["ranked_nodes"],
        1.0,
        float(candidate["sigma_multiplier"]),
        bundle.feature_std,
        int(candidate["trigger_steps"]),
        float(candidate["target_shift_ratio"]),
        float(poison_cfg.get("fallback_shift_ratio", 0.05)),
        ranked_windows=ranking["ranked_windows"],
        window_mode=str(candidate["window_mode"]),
        sample_selection_mode=str(candidate["sample_selection_mode"]),
        target_horizon_mode=str(candidate["target_horizon_mode"]),
        target_horizon_count=int(candidate["target_horizon_count"]),
        target_horizon_indices=ranking.get("target_horizon_indices"),
        target_weight_mode=str(candidate["target_weight_mode"]),
        node_rank_weights=poison_cfg.get("node_rank_weights"),
        tail_horizon_weights=poison_cfg.get("tail_horizon_weights"),
        selection_tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
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

    asr_metrics = compute_attack_success_metrics(
        bundle.test_targets,
        triggered_pred,
        float(candidate["target_shift_ratio"]),
        float(poison_cfg.get("success_tolerance_ratio", 0.03)),
    )
    shift_metrics = compute_prediction_shift_metrics(clean_pred, triggered_pred, float(candidate["target_shift_ratio"]))
    eval_views = compute_attack_evaluation_views(
        bundle.test_targets,
        clean_pred,
        triggered_pred,
        float(candidate["target_shift_ratio"]),
        float(poison_cfg.get("success_tolerance_ratio", 0.03)),
        selected_nodes=poisoned_train["selected_nodes"],
        target_horizon_indices=poisoned_train["selected_target_horizon_indices"],
        tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
        scaler=bundle.scaler,
    )
    stealth_metrics = compute_stealth_metrics(bundle.test_inputs, triggered_inputs)
    clean_deltas = {
        f"clean_{key}": value for key, value in relative_metric_change(baseline_metrics, clean_metrics).items()
    }

    row = {
        "stage_name": stage_name,
        "seed": int(seed),
        "selection_strategy": str(candidate["selection_strategy"]),
        "poison_ratio": float(candidate["poison_ratio"]),
        "sigma_multiplier": float(candidate["sigma_multiplier"]),
        "window_mode": str(candidate["window_mode"]),
        "sample_selection_mode": str(candidate["sample_selection_mode"]),
        "target_weight_mode": str(candidate["target_weight_mode"]),
        "trigger_steps": int(candidate["trigger_steps"]),
        "trigger_node_count": int(candidate["trigger_node_count"]),
        "target_horizon_mode": str(candidate["target_horizon_mode"]),
        "target_horizon_count": int(candidate["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate["time_smoothing_kernel"]),
        "frequency_smoothing_strength": float(candidate["frequency_smoothing_strength"]),
        "frequency_cutoff_ratio": float(candidate["frequency_cutoff_ratio"]),
        "frequency_decay": float(candidate.get("frequency_decay", 0.35)),
        "selected_nodes": ",".join(str(node) for node in poisoned_train["selected_nodes"]),
        "selected_time_indices": ",".join(str(idx) for idx in poisoned_train["selected_time_indices"]),
        "selected_target_horizon_indices": ",".join(str(idx) for idx in poisoned_train["selected_target_horizon_indices"]),
        "node_rank_weights": ",".join(f"{weight:.4f}" for weight in poisoned_train.get("node_rank_weights", [])),
        "tail_horizon_weights": ",".join(f"{weight:.4f}" for weight in poisoned_train.get("tail_horizon_weights", [])),
        "ranked_window_start": int(ranking["ranked_windows"][0]) if ranking["ranked_windows"] else -1,
        "local_forecast_error_mean": float(poisoned_train.get("local_forecast_error_mean", 0.0)),
        "global_forecast_error_mean": float(poisoned_train.get("global_forecast_error_mean", 0.0)),
        "selected_poison_score_mean": float(poisoned_train.get("selected_poison_score_mean", 0.0)),
        **flatten_metrics("baseline", baseline_metrics),
        **flatten_metrics("clean", clean_metrics),
        **flatten_metrics("triggered", triggered_metrics),
        **clean_deltas,
        "attack_success_rate": float(asr_metrics["attack_success_rate"]),
        "target_shift_ratio": float(asr_metrics["target_shift_ratio"]),
        "tolerance_ratio": float(asr_metrics["tolerance_ratio"]),
        **shift_metrics,
        **eval_views,
        **stealth_metrics,
    }

    stealth_row = {
        "stage_name": stage_name,
        "selection_strategy": str(candidate["selection_strategy"]),
        "poison_ratio": float(candidate["poison_ratio"]),
        "sigma_multiplier": float(candidate["sigma_multiplier"]),
        "window_mode": str(candidate["window_mode"]),
        "sample_selection_mode": str(candidate["sample_selection_mode"]),
        "target_weight_mode": str(candidate["target_weight_mode"]),
        "trigger_steps": int(candidate["trigger_steps"]),
        "trigger_node_count": int(candidate["trigger_node_count"]),
        "target_horizon_mode": str(candidate["target_horizon_mode"]),
        "target_horizon_count": int(candidate["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate["time_smoothing_kernel"]),
        "frequency_smoothing_strength": float(candidate["frequency_smoothing_strength"]),
        "frequency_cutoff_ratio": float(candidate["frequency_cutoff_ratio"]),
        "frequency_decay": float(candidate.get("frequency_decay", 0.35)),
        **stealth_metrics,
    }

    payload = None
    if capture_payload:
        payload = {
            "row": row,
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs_from_bundle(config, bundle),
            "baseline_true": baseline_true,
            "baseline_pred": baseline_pred,
            "clean_true": clean_true,
            "clean_pred": clean_pred,
            "triggered_pred": triggered_pred,
            "triggered_inputs": triggered_inputs,
            "selected_nodes": [int(node) for node in poisoned_train["selected_nodes"]],
            "selected_time_indices": [int(idx) for idx in poisoned_train["selected_time_indices"]],
            "selected_target_horizon_indices": [int(idx) for idx in poisoned_train["selected_target_horizon_indices"]],
            "scaler_mean": np.asarray(getattr(bundle.scaler, "mean", np.array([])), dtype=np.float32),
            "scaler_std": np.asarray(getattr(bundle.scaler, "std", np.array([])), dtype=np.float32),
            "artifacts_device": artifacts.device,
        }
    return row, stealth_row, payload


def summarize_recheck_rows(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")

    summary = {
        "stage_name": "recheck_summary",
        "selection_strategy": str(candidate_rows[0]["selection_strategy"]),
        "poison_ratio": float(candidate_rows[0]["poison_ratio"]),
        "sigma_multiplier": float(candidate_rows[0]["sigma_multiplier"]),
        "window_mode": str(candidate_rows[0]["window_mode"]),
        "sample_selection_mode": str(candidate_rows[0].get("sample_selection_mode", "input_energy")),
        "target_weight_mode": str(candidate_rows[0].get("target_weight_mode", "flat")),
        "trigger_steps": int(candidate_rows[0]["trigger_steps"]),
        "trigger_node_count": int(candidate_rows[0]["trigger_node_count"]),
        "target_shift_ratio": float(candidate_rows[0]["target_shift_ratio"]),
        "target_horizon_mode": str(candidate_rows[0]["target_horizon_mode"]),
        "target_horizon_count": int(candidate_rows[0]["target_horizon_count"]),
        "time_smoothing_kernel": int(candidate_rows[0]["time_smoothing_kernel"]),
        "frequency_smoothing_strength": float(candidate_rows[0]["frequency_smoothing_strength"]),
        "frequency_cutoff_ratio": float(candidate_rows[0]["frequency_cutoff_ratio"]),
        "frequency_decay": float(candidate_rows[0].get("frequency_decay", 0.35)),
        "selected_nodes": str(candidate_rows[0]["selected_nodes"]),
        "selected_time_indices": str(candidate_rows[0]["selected_time_indices"]),
        "selected_target_horizon_indices": str(candidate_rows[0].get("selected_target_horizon_indices", "")),
        "ranked_window_start": int(candidate_rows[0]["ranked_window_start"]),
        "recheck_repeat_count": int(len(candidate_rows)),
        "recheck_seed_list": ",".join(str(int(row["seed"])) for row in candidate_rows),
    }

    exclude = {
        "stage_name",
        "selection_strategy",
        "window_mode",
        "sample_selection_mode",
        "target_weight_mode",
        "target_horizon_mode",
        "selected_nodes",
        "selected_time_indices",
        "selected_target_horizon_indices",
        "recheck_repeat_count",
        "recheck_seed_list",
        "seed",
        "candidate_rank",
        "recheck_repeat_index",
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
            "anomaly_rate",
            "target_shift_attainment",
            "mean_prediction_shift_ratio",
            "raw_selected_nodes_tail_horizon_attack_success_rate",
            "raw_selected_nodes_tail_horizon_target_shift_attainment",
            "raw_selected_nodes_attack_success_rate",
            "raw_selected_nodes_target_shift_attainment",
            "raw_global_attack_success_rate",
            "raw_global_target_shift_attainment",
        }:
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def select_top_candidates(
    rows: list[dict[str, Any]],
    max_clean_mae_delta_ratio: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: row_sort_key(item, max_clean_mae_delta_ratio), reverse=True):
        key = candidate_key(candidate_from_row(row))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate_from_row(row))
        if len(deduped) >= top_k:
            break
    return deduped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_poison")

    bundle, baseline_model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, baseline_path = load_or_train_baseline(
        config,
        args.baseline_dir,
    )
    save_json(dataset_summary(bundle, config), run_dir / "dataset_summary.json")

    poison_cfg = config["poison"]
    base_seed = int(config.get("seed", 42))
    max_clean_mae_delta_ratio = float(poison_cfg.get("max_clean_mae_delta_ratio", 0.05))
    search_stages = build_search_stages(poison_cfg)

    all_rows: list[dict[str, Any]] = []
    all_stealth_rows: list[dict[str, Any]] = []
    stage_best_rows: list[dict[str, Any]] = []
    previous_best_row: dict[str, Any] | None = None

    for stage_index, stage_cfg in enumerate(search_stages):
        stage_name = stage_name_from_cfg(stage_cfg, stage_index)
        candidates = resolve_stage_candidates(stage_cfg, poison_cfg, previous_best_row)
        if not candidates:
            raise RuntimeError(f"Search stage '{stage_name}' does not contain any candidate combinations.")

        stage_rows: list[dict[str, Any]] = []
        stage_stealth_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            row, stealth_row, _ = evaluate_attack_candidate(
                config=config,
                bundle=bundle,
                baseline_metrics=baseline_metrics,
                baseline_true=baseline_true,
                baseline_pred=baseline_pred,
                clean_train_pred=clean_train_pred,
                candidate=candidate,
                stage_name=stage_name,
                seed=base_seed,
                capture_payload=False,
            )
            stage_rows.append(row)
            stage_stealth_rows.append(stealth_row)

        stage_best = choose_best_row(stage_rows, max_clean_mae_delta_ratio)
        if stage_best is None:
            raise RuntimeError(f"Search stage '{stage_name}' did not produce any rows.")

        save_table(stage_rows, run_dir / f"{stage_name}_results.csv")
        write_markdown_summary(run_dir / f"{stage_name}_summary.md", f"{stage_name} Results", stage_rows)
        save_json(stage_best, run_dir / f"{stage_name}_best.json")

        all_rows.extend(stage_rows)
        all_stealth_rows.extend(stage_stealth_rows)
        stage_best_rows.append(stage_best)
        previous_best_row = stage_best

    if previous_best_row is None:
        raise RuntimeError("Poisoning sweep did not produce any results.")

    final_stage_name = stage_name_from_cfg(search_stages[-1], len(search_stages) - 1)
    final_stage_rows = [row for row in all_rows if row["stage_name"] == final_stage_name]
    pre_recheck_best = choose_best_row(final_stage_rows, max_clean_mae_delta_ratio)
    if pre_recheck_best is None:
        raise RuntimeError("Final poisoning stage did not produce any rows.")

    recheck_top_k = int(poison_cfg.get("recheck_top_k", 0))
    recheck_repeats = max(1, int(poison_cfg.get("recheck_repeats", 1)))
    recheck_candidates = select_top_candidates(final_stage_rows, max_clean_mae_delta_ratio, recheck_top_k)

    recheck_repeat_rows: list[dict[str, Any]] = []
    recheck_summary_rows: list[dict[str, Any]] = []
    best_repeat_payload_by_candidate: dict[str, dict[str, Any]] = {}
    best_repeat_row_by_candidate: dict[str, dict[str, Any]] = {}

    if recheck_candidates and recheck_repeats > 0:
        for candidate_rank, candidate in enumerate(recheck_candidates, start=1):
            candidate_rows: list[dict[str, Any]] = []
            candidate_best_row: dict[str, Any] | None = None
            candidate_best_payload: dict[str, Any] | None = None

            for repeat_idx in range(recheck_repeats):
                row, _, payload = evaluate_attack_candidate(
                    config=config,
                    bundle=bundle,
                    baseline_metrics=baseline_metrics,
                    baseline_true=baseline_true,
                    baseline_pred=baseline_pred,
                    clean_train_pred=clean_train_pred,
                    candidate=candidate,
                    stage_name="recheck_repeat",
                    seed=base_seed + repeat_idx,
                    capture_payload=True,
                )
                row["candidate_rank"] = int(candidate_rank)
                row["recheck_repeat_index"] = int(repeat_idx)
                candidate_rows.append(row)
                recheck_repeat_rows.append(row)

                if candidate_best_row is None or row_sort_key(row, max_clean_mae_delta_ratio) > row_sort_key(candidate_best_row, max_clean_mae_delta_ratio):
                    candidate_best_row = row
                    candidate_best_payload = payload

            summary_row = summarize_recheck_rows(candidate_rows)
            summary_row["candidate_rank"] = int(candidate_rank)
            recheck_summary_rows.append(summary_row)

            key = candidate_key(candidate)
            if candidate_best_row is not None and candidate_best_payload is not None:
                best_repeat_row_by_candidate[key] = candidate_best_row
                best_repeat_payload_by_candidate[key] = candidate_best_payload

    final_best_row = choose_best_row(recheck_summary_rows, max_clean_mae_delta_ratio) if recheck_summary_rows else None
    final_payload: dict[str, Any] | None = None

    if final_best_row is not None:
        chosen_key = candidate_key(candidate_from_row(final_best_row))
        final_payload = best_repeat_payload_by_candidate.get(chosen_key)
        chosen_repeat_row = best_repeat_row_by_candidate.get(chosen_key)
        if chosen_repeat_row is not None:
            final_best_row = dict(final_best_row)
            final_best_row["selected_repeat_seed"] = int(chosen_repeat_row["seed"])
            final_best_row["selected_repeat_attack_success_rate"] = float(chosen_repeat_row["attack_success_rate"])
            final_best_row["selected_repeat_clean_MAE_delta_ratio"] = float(chosen_repeat_row["clean_MAE_delta_ratio"])
    else:
        _, _, final_payload = evaluate_attack_candidate(
            config=config,
            bundle=bundle,
            baseline_metrics=baseline_metrics,
            baseline_true=baseline_true,
            baseline_pred=baseline_pred,
            clean_train_pred=clean_train_pred,
            candidate=candidate_from_row(pre_recheck_best),
            stage_name="final_export",
            seed=base_seed,
            capture_payload=True,
        )
        final_best_row = pre_recheck_best

    if final_payload is None or final_best_row is None:
        raise RuntimeError("Unable to produce the final poisoning payload.")

    save_table(all_rows, run_dir / "attack_results.csv")
    save_table(all_stealth_rows, run_dir / "stealth_results.csv")
    save_table(
        [
            {
                "stage_name": row["stage_name"],
                "selection_strategy": row["selection_strategy"],
                "poison_ratio": row["poison_ratio"],
                "sigma_multiplier": row["sigma_multiplier"],
                "target_shift_ratio": row["target_shift_ratio"],
                "window_mode": row["window_mode"],
                "sample_selection_mode": row["sample_selection_mode"],
                "target_weight_mode": row["target_weight_mode"],
                "trigger_steps": row["trigger_steps"],
                "trigger_node_count": row["trigger_node_count"],
                "attack_success_rate": row["attack_success_rate"],
                "raw_selected_nodes_tail_horizon_attack_success_rate": row.get("raw_selected_nodes_tail_horizon_attack_success_rate"),
                "raw_selected_nodes_attack_success_rate": row.get("raw_selected_nodes_attack_success_rate"),
                "raw_global_attack_success_rate": row.get("raw_global_attack_success_rate"),
                "clean_MAE_delta_ratio": row["clean_MAE_delta_ratio"],
                "local_forecast_error_mean": row.get("local_forecast_error_mean"),
                "global_forecast_error_mean": row.get("global_forecast_error_mean"),
                "selected_poison_score_mean": row.get("selected_poison_score_mean"),
                "raw_selected_nodes_tail_horizon_target_shift_attainment": row.get("raw_selected_nodes_tail_horizon_target_shift_attainment"),
                "anomaly_rate": row["anomaly_rate"],
            }
            for row in all_rows
        ],
        run_dir / "ablation_table.csv",
    )
    write_markdown_summary(run_dir / "attack_summary.md", "Attack Results", all_rows)

    if recheck_repeat_rows:
        save_table(recheck_repeat_rows, run_dir / "recheck_repeats.csv")
    if recheck_summary_rows:
        save_table(recheck_summary_rows, run_dir / "recheck_results.csv")
        write_markdown_summary(run_dir / "recheck_summary.md", "Recheck Results", recheck_summary_rows)

    save_json(pre_recheck_best, run_dir / "best_attack_pre_recheck.json")
    save_json(final_best_row, run_dir / "best_attack.json")
    save_json(
        {
            "stage_count": len(search_stages),
            "stage_names": [stage_name_from_cfg(stage_cfg, idx) for idx, stage_cfg in enumerate(search_stages)],
            "stage_bests": stage_best_rows,
            "pre_recheck_best": pre_recheck_best,
            "final_best": final_best_row,
            "recheck_candidate_count": len(recheck_candidates),
            "recheck_repeat_count": recheck_repeats if recheck_candidates else 0,
        },
        run_dir / "search_summary.json",
    )

    torch.save(
        {
            "config": config,
            "model_kwargs": final_payload["model_kwargs"],
            "model_state": final_payload["model_state"],
            "best_row": final_best_row,
            "pre_recheck_best_row": pre_recheck_best,
        },
        run_dir / "best_poisoned_model.pt",
    )
    np.savez(
        run_dir / "best_attack_bundle.npz",
        clean_test_inputs=bundle.test_inputs,
        triggered_test_inputs=final_payload["triggered_inputs"],
        test_targets=bundle.test_targets,
        baseline_test_predictions=baseline_pred,
        poisoned_model_clean_predictions=final_payload["clean_pred"],
        poisoned_model_trigger_predictions=final_payload["triggered_pred"],
        selected_nodes=np.asarray(final_payload["selected_nodes"], dtype=np.int64),
        selected_time_indices=np.asarray(final_payload["selected_time_indices"], dtype=np.int64),
        selected_target_horizon_indices=np.asarray(final_payload["selected_target_horizon_indices"], dtype=np.int64),
        scaler_mean=np.asarray(final_payload["scaler_mean"], dtype=np.float32),
        scaler_std=np.asarray(final_payload["scaler_std"], dtype=np.float32),
    )

    plot_prediction_case(
        baseline_true,
        baseline_pred,
        run_dir / "best_prediction_case.png",
        poisoned_pred=final_payload["triggered_pred"],
        sample_index=0,
        node_index=0,
        title="Baseline vs Triggered Prediction",
    )
    plot_trigger_case(
        bundle.test_inputs,
        final_payload["triggered_inputs"],
        run_dir / "trigger_case.png",
        sample_index=0,
        node_indices=list(final_payload["selected_nodes"]),
    )

    save_json(
        {
            "baseline_dir": str(baseline_path) if baseline_path else None,
            "best_attack_path": str((run_dir / "best_attack.json").resolve()),
            "best_model_path": str((run_dir / "best_poisoned_model.pt").resolve()),
            "search_summary_path": str((run_dir / "search_summary.json").resolve()),
            "recheck_results_path": str((run_dir / "recheck_results.csv").resolve()) if recheck_summary_rows else None,
        },
        run_dir / "run_manifest.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
