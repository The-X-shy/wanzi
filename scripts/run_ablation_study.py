"""Ablation study: vary one dimension at a time while holding others fixed.

Usage:
    python scripts/run_ablation_study.py --config configs/ablation_poison_ratio.yaml
"""

from __future__ import annotations

import argparse
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
    build_model, evaluate_on_arrays, make_loader, model_kwargs_from_bundle,
    prepare_bundle_from_config, relative_metric_change, write_markdown_summary,
)
from traffic_poison.poisoning import (
    build_poisoned_training_set, compute_attack_evaluation_views,
    compute_stealth_metrics, rank_vulnerable_positions,
)
from traffic_poison.reporting import plot_bar_table, save_table
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, flatten_metrics, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study over one parameter.")
    parser.add_argument("--config", required=True, help="Ablation config YAML.")
    parser.add_argument("--baseline-dir", help="Clean baseline directory.")
    return parser.parse_args()


def load_or_train_baseline_quick(config: dict[str, Any], baseline_dir: str | None):
    from traffic_poison.experiment import build_model_from_kwargs

    bundle = prepare_bundle_from_config(config, seed=int(config.get("seed", 42)))
    if baseline_dir:
        baseline_path = Path(baseline_dir).resolve()
        checkpoint = torch.load(baseline_path / "clean_model.pt", map_location="cpu")
        model = build_model_from_kwargs(checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        _, _, clean_train_pred = evaluate_on_arrays(
            model, bundle.train_inputs, bundle.train_targets, config, "cpu",
        )
        baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device="cpu")
        return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred

    set_seed(int(config.get("seed", 42)))
    model = build_model(config, bundle)
    artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])
    baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
    _, _, clean_train_pred = evaluate_on_arrays(model, bundle.train_inputs, bundle.train_targets, config, artifacts.device)
    return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ablation_cfg = config.get("ablation", {})
    vary_param = str(ablation_cfg["vary_param"])
    vary_values = list(ablation_cfg["vary_values"])
    n_repeats = int(ablation_cfg.get("repeats", 3))
    seed = int(config.get("seed", 42))
    run_dir = create_run_dir(config["output"]["root_dir"], f"ablation_{vary_param}")

    bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred = load_or_train_baseline_quick(
        config, args.baseline_dir,
    )

    poison_cfg = config["poison"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    results: list[dict[str, Any]] = []

    for value in vary_values:
        for repeat in range(n_repeats):
            set_seed(seed + repeat)
            candidate = {
                "selection_strategy": str(poison_cfg.get("selection_strategies", ["error"])[0]),
                "poison_ratio": float(poison_cfg.get("poison_ratios", [0.02])[0]),
                "sigma_multiplier": float(poison_cfg.get("sigma_multipliers", [0.065])[0]),
                "target_shift_ratio": float(poison_cfg.get("target_shift_ratios", [0.075])[0]),
                "window_mode": str(poison_cfg.get("window_modes", ["hybrid"])[0]),
                "sample_selection_mode": str(poison_cfg.get("sample_selection_modes", ["directional_headroom"])[0]),
                "target_weight_mode": str(poison_cfg.get("target_weight_modes", ["dual_focus"])[0]),
                "trigger_steps": int(poison_cfg.get("trigger_steps", 3)),
                "trigger_node_count": int(poison_cfg.get("trigger_node_count", 3)),
                "target_horizon_mode": str(poison_cfg.get("target_horizon_modes", ["all"])[0]),
                "target_horizon_count": int(poison_cfg.get("target_horizon_count", 3)),
                "time_smoothing_kernel": int(poison_cfg.get("time_smoothing_kernels", [3])[0]),
                "frequency_smoothing_strength": float(poison_cfg.get("frequency_smoothing_strengths", [0.05])[0]),
                "frequency_cutoff_ratio": float(poison_cfg.get("frequency_cutoff_ratios", [0.5])[0]),
                "frequency_decay": float(poison_cfg.get("frequency_decays", [0.35])[0]),
                "spectral_constraint_strength": float(poison_cfg.get("spectral_constraint_strengths", [0.0])[0]),
                "headroom_error_mix": float(poison_cfg.get("headroom_error_mixes", [0.6])[0]),
                "global_shift_fraction": float(poison_cfg.get("global_shift_fractions", [0.3])[0]),
                "tail_focus_multiplier": float(poison_cfg.get("tail_focus_multipliers", [1.6])[0]),
                "loss_focus_mode": str(poison_cfg.get("loss_focus_modes", ["directional_focus"])[0]),
                "loss_selected_node_weight": float(poison_cfg.get("loss_selected_node_weights", [1.2])[0]),
                "loss_tail_horizon_weight": float(poison_cfg.get("loss_tail_horizon_weights", [1.9])[0]),
                "loss_headroom_boost": float(poison_cfg.get("loss_headroom_boosts", [0.6])[0]),
            }
            # Override the parameter being varied
            candidate[vary_param] = value

            ranking = rank_vulnerable_positions(
                bundle.train_inputs, bundle.train_targets, clean_train_pred,
                bundle.adjacency, candidate["selection_strategy"],
                int(candidate["trigger_node_count"]), int(candidate["trigger_steps"]),
                target_horizon_count=int(candidate["target_horizon_count"]),
                target_horizon_mode=str(candidate["target_horizon_mode"]),
            )

            poisoned_train = build_poisoned_training_set(
                bundle.train_inputs, bundle.train_targets,
                ranking["ranked_nodes"],
                float(candidate["poison_ratio"]), float(candidate["sigma_multiplier"]),
                bundle.feature_std, int(candidate["trigger_steps"]),
                float(candidate["target_shift_ratio"]),
                float(poison_cfg.get("fallback_shift_ratio", 0.05)),
                ranked_windows=ranking["ranked_windows"],
                window_mode=str(candidate["window_mode"]),
                sample_selection_mode=str(candidate["sample_selection_mode"]),
                target_weight_mode=str(candidate["target_weight_mode"]),
                target_horizon_mode=str(candidate["target_horizon_mode"]),
                target_horizon_count=int(candidate["target_horizon_count"]),
                target_horizon_indices=ranking.get("target_horizon_indices"),
                clean_predictions=clean_train_pred,
                node_rank_weights=poison_cfg.get("node_rank_weights"),
                tail_horizon_weights=poison_cfg.get("tail_horizon_weights"),
                selection_tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
                time_smoothing_kernel=int(candidate["time_smoothing_kernel"]),
                frequency_smoothing_strength=float(candidate["frequency_smoothing_strength"]),
                frequency_cutoff_ratio=float(candidate["frequency_cutoff_ratio"]),
                frequency_decay=float(candidate.get("frequency_decay", 0.35)),
                headroom_floor=float(poison_cfg.get("headroom_floor", 0.0)),
                headroom_error_mix=float(candidate.get("headroom_error_mix", 0.6)),
                global_shift_fraction=float(candidate.get("global_shift_fraction", 0.3)),
                tail_focus_multiplier=float(candidate.get("tail_focus_multiplier", 1.6)),
                loss_focus_mode=str(candidate.get("loss_focus_mode", "directional_focus")),
                loss_selected_node_weight=float(candidate.get("loss_selected_node_weight", 1.2)),
                loss_tail_horizon_weight=float(candidate.get("loss_tail_horizon_weight", 1.9)),
                loss_headroom_boost=float(candidate.get("loss_headroom_boost", 0.6)),
                feature_scaler=bundle.scaler,
                trigger_feature_std=bundle.feature_std,
                spectral_constraint_strength=float(candidate.get("spectral_constraint_strength", 0.0)),
            )

            poisoned_loader = make_loader(
                np.asarray(poisoned_train["poisoned_inputs"]),
                np.asarray(poisoned_train["poisoned_targets"]),
                loss_weights=np.asarray(poisoned_train["poisoned_loss_weights"]),
                batch_size=int(dataset_cfg.get("batch_size", 64)), shuffle=True,
            )

            model = build_model(config, bundle)
            artifacts = train_model(model, poisoned_loader, bundle.val_loader, training_cfg)
            clean_metrics, clean_true, clean_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)

            triggered_test = build_poisoned_training_set(
                bundle.test_inputs, bundle.test_targets,
                ranking["ranked_nodes"], 1.0,
                float(candidate["sigma_multiplier"]), bundle.feature_std,
                int(candidate["trigger_steps"]), float(candidate["target_shift_ratio"]),
                float(poison_cfg.get("fallback_shift_ratio", 0.05)),
                ranked_windows=ranking["ranked_windows"],
                window_mode=str(candidate["window_mode"]),
                sample_selection_mode=str(candidate["sample_selection_mode"]),
                target_weight_mode=str(candidate["target_weight_mode"]),
                target_horizon_mode=str(candidate["target_horizon_mode"]),
                target_horizon_count=int(candidate["target_horizon_count"]),
                target_horizon_indices=ranking.get("target_horizon_indices"),
                node_rank_weights=poison_cfg.get("node_rank_weights"),
                tail_horizon_weights=poison_cfg.get("tail_horizon_weights"),
                selection_tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
                time_smoothing_kernel=int(candidate["time_smoothing_kernel"]),
                frequency_smoothing_strength=float(candidate["frequency_smoothing_strength"]),
                frequency_cutoff_ratio=float(candidate["frequency_cutoff_ratio"]),
                frequency_decay=float(candidate.get("frequency_decay", 0.35)),
                headroom_floor=float(poison_cfg.get("headroom_floor", 0.0)),
                headroom_error_mix=float(candidate.get("headroom_error_mix", 0.6)),
                global_shift_fraction=float(candidate.get("global_shift_fraction", 0.3)),
                tail_focus_multiplier=float(candidate.get("tail_focus_multiplier", 1.6)),
                loss_focus_mode=str(candidate.get("loss_focus_mode", "directional_focus")),
                loss_selected_node_weight=float(candidate.get("loss_selected_node_weight", 1.2)),
                loss_tail_horizon_weight=float(candidate.get("loss_tail_horizon_weight", 1.9)),
                loss_headroom_boost=float(candidate.get("loss_headroom_boost", 0.6)),
                feature_scaler=bundle.scaler,
                trigger_feature_std=bundle.feature_std,
                spectral_constraint_strength=float(candidate.get("spectral_constraint_strength", 0.0)),
            )
            triggered_inputs = np.asarray(triggered_test["poisoned_inputs"])
            _, _, triggered_pred = evaluate_on_arrays(model, triggered_inputs, bundle.test_targets, config, artifacts.device)

            eval_views = compute_attack_evaluation_views(
                bundle.test_targets, clean_pred, triggered_pred,
                float(candidate["target_shift_ratio"]),
                float(poison_cfg.get("success_tolerance_ratio", 0.03)),
                selected_nodes=poisoned_train["selected_nodes"],
                target_horizon_indices=poisoned_train["selected_target_horizon_indices"],
                tail_horizon_count=int(poison_cfg.get("evaluation_tail_horizon_count", 3)),
                scaler=bundle.scaler,
            )
            stealth = compute_stealth_metrics(bundle.test_inputs, triggered_inputs)
            clean_deltas = {f"clean_{k}": v for k, v in relative_metric_change(baseline_metrics, clean_metrics).items()}

            row = {
                "vary_param": vary_param,
                "vary_value": value,
                "repeat": repeat,
                "seed": int(seed + repeat),
                **flatten_metrics("clean", clean_metrics),
                **clean_deltas,
                "attack_success_rate": float(eval_views["raw_global_attack_success_rate"]),
                "local_tail_asr": float(eval_views.get("raw_selected_nodes_tail_horizon_attack_success_rate", 0.0)),
                "direction_match_rate": float(eval_views.get("raw_selected_nodes_tail_horizon_shift_direction_match_rate", 0.0)),
                "target_shift_attainment": float(eval_views.get("raw_selected_nodes_tail_horizon_target_shift_attainment", 0.0)),
                **stealth,
            }
            results.append(row)

    save_table(results, run_dir / "ablation_results.csv")
    write_markdown_summary(run_dir / "ablation_summary.md", f"Ablation: {vary_param}", results)

    # Compute aggregated stats per value
    agg_rows: list[dict[str, Any]] = []
    for value in vary_values:
        subset = [r for r in results if r["vary_value"] == value]
        if not subset:
            continue
        agg = {"vary_param": vary_param, "vary_value": value, "n_repeats": len(subset)}
        for key in ["clean_MAE_delta_ratio", "attack_success_rate", "local_tail_asr",
                     "direction_match_rate", "target_shift_attainment",
                     "frequency_energy_shift", "mean_z_score"]:
            vals = [float(r.get(key, 0.0)) for r in subset]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        agg_rows.append(agg)

    save_table(agg_rows, run_dir / "ablation_aggregated.csv")
    save_json(agg_rows, run_dir / "ablation_aggregated.json")

    plot_bar_table(results, "vary_value", "local_tail_asr", run_dir / "ablation_local_asr.png",
                   f"Ablation: {vary_param} vs Local ASR")
    plot_bar_table(results, "vary_value", "clean_MAE_delta_ratio", run_dir / "ablation_clean_mae.png",
                   f"Ablation: {vary_param} vs Clean MAE Drift")

    print(run_dir)


if __name__ == "__main__":
    main()
