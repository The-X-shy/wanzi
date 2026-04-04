from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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
    compute_attack_success_metrics,
    compute_stealth_metrics,
    rank_vulnerable_positions,
)
from traffic_poison.reporting import plot_prediction_case
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the best attack settings on a second dataset.")
    parser.add_argument("--config", required=True, help="Path to the secondary dataset config, usually PEMS-BAY.")
    parser.add_argument("--best-attack-json", required=True, help="Path to best_attack.json from the main poisoning run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    with Path(args.best_attack_json).resolve().open("r", encoding="utf-8") as handle:
        best_attack = json.load(handle)

    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_cross")
    set_seed(int(config.get("seed", 42)))

    bundle = prepare_bundle_from_config(config)
    model = build_model(config, bundle)
    clean_artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])
    clean_metrics, clean_true, clean_pred = evaluate_model(model, bundle.test_loader, device=clean_artifacts.device)
    _, _, clean_train_pred = evaluate_on_arrays(model, bundle.train_inputs, bundle.train_targets, config, clean_artifacts.device)

    ranking = rank_vulnerable_positions(
        bundle.train_inputs,
        bundle.train_targets,
        clean_train_pred,
        bundle.adjacency,
        best_attack["selection_strategy"],
        int(config["poison"].get("trigger_node_count", 5)),
        int(config["poison"].get("trigger_steps", 3)),
    )
    poisoned_train = build_poisoned_training_set(
        bundle.train_inputs,
        bundle.train_targets,
        ranking["ranked_nodes"],
        float(best_attack["poison_ratio"]),
        float(best_attack["sigma_multiplier"]),
        bundle.feature_std,
        int(config["poison"].get("trigger_steps", 3)),
        float(config["poison"].get("target_shift_ratio", 0.10)),
        float(config["poison"].get("fallback_shift_ratio", 0.05)),
    )
    poisoned_loader = make_loader(
        poisoned_train["poisoned_inputs"],
        poisoned_train["poisoned_targets"],
        batch_size=int(config["dataset"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config["dataset"].get("num_workers", 0)),
    )

    poisoned_model = build_model(config, bundle)
    poison_artifacts = train_model(poisoned_model, poisoned_loader, bundle.val_loader, config["training"])
    poisoned_clean_metrics, _, _ = evaluate_model(poisoned_model, bundle.test_loader, device=poison_artifacts.device)

    triggered_test = build_poisoned_training_set(
        bundle.test_inputs,
        bundle.test_targets,
        ranking["ranked_nodes"],
        1.0,
        float(best_attack["sigma_multiplier"]),
        bundle.feature_std,
        int(config["poison"].get("trigger_steps", 3)),
        float(config["poison"].get("target_shift_ratio", 0.10)),
        float(config["poison"].get("fallback_shift_ratio", 0.05)),
    )
    triggered_metrics, _, triggered_pred = evaluate_on_arrays(
        poisoned_model,
        triggered_test["poisoned_inputs"],
        bundle.test_targets,
        config,
        poison_artifacts.device,
    )
    asr = compute_attack_success_metrics(
        bundle.test_targets,
        triggered_pred,
        float(config["poison"].get("target_shift_ratio", 0.10)),
        float(config["poison"].get("success_tolerance_ratio", 0.03)),
    )
    stealth = compute_stealth_metrics(bundle.test_inputs, triggered_test["poisoned_inputs"])

    summary = {
        "dataset_summary": dataset_summary(bundle, config),
        "best_attack_replayed": best_attack,
        "clean_metrics": clean_metrics,
        "poisoned_clean_metrics": poisoned_clean_metrics,
        "triggered_metrics": triggered_metrics,
        "attack_success_rate": float(asr["attack_success_rate"]),
        "clean_metric_change": relative_metric_change(clean_metrics, poisoned_clean_metrics),
        "stealth_metrics": stealth,
    }

    save_json(summary, run_dir / "cross_dataset_summary.json")
    plot_prediction_case(
        clean_true,
        clean_pred,
        run_dir / "cross_dataset_prediction_case.png",
        poisoned_pred=triggered_pred,
        sample_index=0,
        node_index=0,
        title="Cross-Dataset Replay",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
