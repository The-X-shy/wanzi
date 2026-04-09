from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


DEFAULT_THESIS_CONTRACT: dict[str, Any] = {
    "main_local_asr_key": "raw_selected_nodes_tail_horizon_attack_success_rate",
    "secondary_local_asr_key": "raw_selected_nodes_attack_success_rate",
    "legacy_asr_key": "attack_success_rate",
    "shift_direction_match_rate_key": "raw_selected_nodes_tail_horizon_shift_direction_match_rate",
    "target_shift_attainment_key": "raw_selected_nodes_tail_horizon_target_shift_attainment",
    "frequency_energy_shift_key": "frequency_energy_shift",
    "mean_z_score_key": "mean_z_score",
    "baseline_best_mae_max": 0.37,
    "baseline_best_mae_strong_max": 0.366,
    "baseline_spread_max": 0.05,
    "baseline_spread_strong_max": 0.035,
    "clean_mae_delta_ratio_max": 0.05,
    "clean_mae_delta_ratio_strong_max": 0.04,
    "minimum_main_local_asr": 0.05,
    "strong_main_local_asr": 0.06,
    "minimum_legacy_asr": 0.015,
    "strong_legacy_asr": 0.018,
    "minimum_shift_direction_match_rate": 0.60,
    "strong_shift_direction_match_rate": 0.65,
    "strong_target_shift_attainment_min": 0.0,
    "maximum_frequency_energy_shift": 0.05,
    "strong_frequency_energy_shift": 0.045,
    "maximum_mean_z_score": 0.80,
    "strong_mean_z_score": 0.75,
    "minimum_cross_local_asr": 0.06,
    "strong_cross_local_asr": 0.07,
    "cross_clean_mae_delta_ratio_max": 0.05,
    "cross_clean_mae_delta_ratio_strong_max": 0.03,
    "previous_mainline_local_asr": 0.056059718969555035,
    "cross_replay_clean_mae_delta_ratio_max": 0.04,
    "cross_replay_target_shift_attainment_min": -0.001,
}


def resolve_thesis_contract(source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    contract = deepcopy(DEFAULT_THESIS_CONTRACT)
    if source is None:
        return contract
    override = source.get("thesis_contract", source) if isinstance(source, Mapping) else {}
    for key, value in dict(override).items():
        contract[key] = value
    return contract


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_first_float(row: Mapping[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return _as_float(row[key], default)
    return default


def main_local_asr(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(
        row,
        [
            str(thesis_contract["main_local_asr_key"]),
            str(thesis_contract.get("secondary_local_asr_key", "raw_selected_nodes_attack_success_rate")),
            "raw_global_attack_success_rate",
            str(thesis_contract.get("legacy_asr_key", "attack_success_rate")),
        ],
        default=0.0,
    )


def secondary_local_asr(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(
        row,
        [
            str(thesis_contract.get("secondary_local_asr_key", "raw_selected_nodes_attack_success_rate")),
            "raw_global_attack_success_rate",
            str(thesis_contract.get("legacy_asr_key", "attack_success_rate")),
        ],
        default=0.0,
    )


def legacy_asr(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(row, [str(thesis_contract.get("legacy_asr_key", "attack_success_rate"))], default=0.0)


def clean_mae_delta_ratio(row: Mapping[str, Any]) -> float:
    return _get_first_float(row, ["clean_MAE_delta_ratio", "MAE_delta_ratio", "clean_delta_ratio"], default=float("inf"))


def shift_direction_match_rate(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(
        row,
        [
            str(thesis_contract["shift_direction_match_rate_key"]),
            "raw_selected_nodes_shift_direction_match_rate",
            "raw_global_shift_direction_match_rate",
            "shift_direction_match_rate",
        ],
        default=0.0,
    )


def target_shift_attainment(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(
        row,
        [
            str(thesis_contract["target_shift_attainment_key"]),
            "raw_selected_nodes_target_shift_attainment",
            "raw_global_target_shift_attainment",
            "target_shift_attainment",
        ],
        default=0.0,
    )


def frequency_energy_shift(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(
        row,
        [str(thesis_contract["frequency_energy_shift_key"]), "anomaly_rate"],
        default=0.0,
    )


def mean_z_score(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _get_first_float(row, [str(thesis_contract["mean_z_score_key"])], default=0.0)


def anomaly_rate(row: Mapping[str, Any]) -> float:
    return _get_first_float(row, ["anomaly_rate"], default=0.0)


def within_clean_budget(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    if row is None:
        return False
    return clean_mae_delta_ratio(row) <= _as_float(thesis_contract.get("clean_mae_delta_ratio_max"), 0.05)


def direction_ok(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    if row is None:
        return False
    return shift_direction_match_rate(row, thesis_contract) >= _as_float(
        thesis_contract.get("minimum_shift_direction_match_rate"),
        0.60,
    )


def strong_direction_ok(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    if row is None:
        return False
    return (
        shift_direction_match_rate(row, thesis_contract)
        >= _as_float(thesis_contract.get("strong_shift_direction_match_rate"), 0.65)
        and target_shift_attainment(row, thesis_contract)
        >= _as_float(thesis_contract.get("strong_target_shift_attainment_min"), 0.0)
    )


def passes_candidate_contract(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    return within_clean_budget(row, thesis_contract) and direction_ok(row, thesis_contract)


def _shortfall(actual: float, minimum: float) -> float:
    return max(0.0, minimum - actual)


def _overflow(actual: float, maximum: float) -> float:
    return max(0.0, actual - maximum)


def legacy_asr_gap_to_minimum(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _shortfall(legacy_asr(row, thesis_contract), _as_float(thesis_contract.get("minimum_legacy_asr"), 0.015))


def main_local_asr_gap_to_minimum(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _shortfall(main_local_asr(row, thesis_contract), _as_float(thesis_contract.get("minimum_main_local_asr"), 0.05))


def shift_direction_gap_to_minimum(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _shortfall(
        shift_direction_match_rate(row, thesis_contract),
        _as_float(thesis_contract.get("minimum_shift_direction_match_rate"), 0.60),
    )


def clean_budget_overflow(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    return _overflow(clean_mae_delta_ratio(row), _as_float(thesis_contract.get("clean_mae_delta_ratio_max"), 0.05))


def stealth_overflow(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> float:
    frequency_gap = _overflow(
        frequency_energy_shift(row, thesis_contract),
        _as_float(thesis_contract.get("maximum_frequency_energy_shift"), 0.05),
    )
    z_gap = _overflow(mean_z_score(row, thesis_contract), _as_float(thesis_contract.get("maximum_mean_z_score"), 0.80))
    return max(frequency_gap, z_gap)


def passes_minimum_candidate_bar(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    return passes_main_result_minimum_bar(row, None, None, thesis_contract)


def passes_strong_candidate_bar(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    return passes_main_result_strong_bar(row, None, None, thesis_contract)


def raw_score_candidate(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        main_local_asr(row, thesis_contract),
        target_shift_attainment(row, thesis_contract),
        legacy_asr(row, thesis_contract),
        1.0 if within_clean_budget(row, thesis_contract) else 0.0,
        1.0 if direction_ok(row, thesis_contract) else 0.0,
        -frequency_energy_shift(row, thesis_contract),
        -abs(clean_mae_delta_ratio(row)),
        secondary_local_asr(row, thesis_contract),
        -mean_z_score(row, thesis_contract),
        -anomaly_rate(row),
    )


def _paper_pass_score(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        main_local_asr(row, thesis_contract),
        target_shift_attainment(row, thesis_contract),
        -frequency_energy_shift(row, thesis_contract),
        -abs(clean_mae_delta_ratio(row)),
        legacy_asr(row, thesis_contract),
        secondary_local_asr(row, thesis_contract),
        -mean_z_score(row, thesis_contract),
        -anomaly_rate(row),
    )


def _paper_gap_score(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -legacy_asr_gap_to_minimum(row, thesis_contract),
        -main_local_asr_gap_to_minimum(row, thesis_contract),
        -shift_direction_gap_to_minimum(row, thesis_contract),
        -clean_budget_overflow(row, thesis_contract),
        -stealth_overflow(row, thesis_contract),
        main_local_asr(row, thesis_contract),
        target_shift_attainment(row, thesis_contract),
        legacy_asr(row, thesis_contract),
        -frequency_energy_shift(row, thesis_contract),
        -abs(clean_mae_delta_ratio(row)),
        secondary_local_asr(row, thesis_contract),
        -mean_z_score(row, thesis_contract),
        -anomaly_rate(row),
    )


def paper_score_candidate(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    minimum_pass = passes_minimum_candidate_bar(row, thesis_contract)
    return (
        1.0 if within_clean_budget(row, thesis_contract) else 0.0,
        1.0 if direction_ok(row, thesis_contract) else 0.0,
        1.0 if minimum_pass else 0.0,
        *(_paper_pass_score(row, thesis_contract) if minimum_pass else _paper_gap_score(row, thesis_contract)),
    )


def score_candidate(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return paper_score_candidate(row, thesis_contract)


def row_sort_key(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return paper_score_candidate(row, thesis_contract)


def raw_row_sort_key(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> tuple[float, ...]:
    return raw_score_candidate(row, thesis_contract)


def choose_best_row(rows: list[dict[str, Any]], thesis_contract: Mapping[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row_sort_key(row, thesis_contract))


def choose_best_paper_row(rows: list[dict[str, Any]], thesis_contract: Mapping[str, Any]) -> dict[str, Any] | None:
    return choose_best_row(rows, thesis_contract)


def choose_best_raw_row(rows: list[dict[str, Any]], thesis_contract: Mapping[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: raw_row_sort_key(row, thesis_contract))


def candidate_contract_flags(row: Mapping[str, Any], thesis_contract: Mapping[str, Any]) -> dict[str, Any]:
    within_budget = within_clean_budget(row, thesis_contract)
    return {
        "within_clean_budget": within_budget,
        "within_budget": within_budget,
        "direction_ok": direction_ok(row, thesis_contract),
        "strong_direction_ok": strong_direction_ok(row, thesis_contract),
        "minimum_contract_pass": passes_minimum_candidate_bar(row, thesis_contract),
        "strong_contract_pass": passes_strong_candidate_bar(row, thesis_contract),
        "eligible_for_cross_replay": eligible_for_cross_replay(row, thesis_contract),
        "legacy_asr_gap_to_minimum": legacy_asr_gap_to_minimum(row, thesis_contract),
        "main_local_asr_gap_to_minimum": main_local_asr_gap_to_minimum(row, thesis_contract),
        "shift_direction_gap_to_minimum": shift_direction_gap_to_minimum(row, thesis_contract),
        "clean_budget_overflow": clean_budget_overflow(row, thesis_contract),
        "stealth_overflow": stealth_overflow(row, thesis_contract),
    }


def eligible_for_cross_replay(row: Mapping[str, Any] | None, thesis_contract: Mapping[str, Any]) -> bool:
    if row is None:
        return False
    return (
        main_local_asr(row, thesis_contract)
        > _as_float(thesis_contract.get("previous_mainline_local_asr"), 0.0)
        and clean_mae_delta_ratio(row)
        <= _as_float(thesis_contract.get("cross_replay_clean_mae_delta_ratio_max"), 0.04)
        and direction_ok(row, thesis_contract)
        and target_shift_attainment(row, thesis_contract)
        > _as_float(thesis_contract.get("cross_replay_target_shift_attainment_min"), -0.001)
    )


def passes_main_result_minimum_bar(
    final_best: Mapping[str, Any] | None,
    baseline_best_mae: float | None,
    baseline_spread: float | None,
    thesis_contract: Mapping[str, Any],
) -> bool:
    if final_best is None:
        return False
    if baseline_best_mae is not None and baseline_best_mae > _as_float(thesis_contract.get("baseline_best_mae_max"), 0.37):
        return False
    if baseline_spread is not None and baseline_spread > _as_float(thesis_contract.get("baseline_spread_max"), 0.05):
        return False
    return (
        clean_mae_delta_ratio(final_best) <= _as_float(thesis_contract.get("clean_mae_delta_ratio_max"), 0.05)
        and main_local_asr(final_best, thesis_contract) >= _as_float(thesis_contract.get("minimum_main_local_asr"), 0.05)
        and legacy_asr(final_best, thesis_contract) >= _as_float(thesis_contract.get("minimum_legacy_asr"), 0.015)
        and shift_direction_match_rate(final_best, thesis_contract)
        >= _as_float(thesis_contract.get("minimum_shift_direction_match_rate"), 0.60)
        and frequency_energy_shift(final_best, thesis_contract)
        <= _as_float(thesis_contract.get("maximum_frequency_energy_shift"), 0.05)
        and mean_z_score(final_best, thesis_contract) <= _as_float(thesis_contract.get("maximum_mean_z_score"), 0.80)
    )


def passes_main_result_strong_bar(
    final_best: Mapping[str, Any] | None,
    baseline_best_mae: float | None,
    baseline_spread: float | None,
    thesis_contract: Mapping[str, Any],
) -> bool:
    if final_best is None:
        return False
    if baseline_best_mae is not None and baseline_best_mae > _as_float(thesis_contract.get("baseline_best_mae_strong_max"), 0.366):
        return False
    if baseline_spread is not None and baseline_spread > _as_float(thesis_contract.get("baseline_spread_strong_max"), 0.035):
        return False
    return (
        clean_mae_delta_ratio(final_best)
        <= _as_float(thesis_contract.get("clean_mae_delta_ratio_strong_max"), 0.04)
        and main_local_asr(final_best, thesis_contract) >= _as_float(thesis_contract.get("strong_main_local_asr"), 0.06)
        and legacy_asr(final_best, thesis_contract) >= _as_float(thesis_contract.get("strong_legacy_asr"), 0.018)
        and strong_direction_ok(final_best, thesis_contract)
        and frequency_energy_shift(final_best, thesis_contract)
        <= _as_float(thesis_contract.get("strong_frequency_energy_shift"), 0.045)
        and mean_z_score(final_best, thesis_contract) <= _as_float(thesis_contract.get("strong_mean_z_score"), 0.75)
    )


def evaluate_main_result_standards(
    final_best: Mapping[str, Any] | None,
    baseline_best_mae: float | None,
    baseline_spread: float | None,
    thesis_contract: Mapping[str, Any],
) -> dict[str, Any]:
    previous_bar = _as_float(thesis_contract.get("previous_mainline_local_asr"), 0.0)
    final_local_asr = main_local_asr(final_best or {}, thesis_contract)
    final_target_shift = target_shift_attainment(final_best or {}, thesis_contract)
    return {
        "minimum_bar_met": passes_main_result_minimum_bar(final_best, baseline_best_mae, baseline_spread, thesis_contract),
        "strong_bar_met": passes_main_result_strong_bar(final_best, baseline_best_mae, baseline_spread, thesis_contract),
        "beat_previous_local_mainline": final_local_asr > previous_bar,
        "stop_rule_triggered": final_local_asr <= previous_bar or final_target_shift < _as_float(
            thesis_contract.get("strong_target_shift_attainment_min"),
            0.0,
        ),
        "follow_up_search_recommended": eligible_for_cross_replay(final_best, thesis_contract),
    }


def evaluate_cross_result_standards(
    cross_best: Mapping[str, Any] | None,
    thesis_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if cross_best is None:
        return {
            "cross_minimum_bar_met": False,
            "cross_strong_bar_met": False,
        }
    return {
        "cross_minimum_bar_met": (
            main_local_asr(cross_best, thesis_contract) >= _as_float(thesis_contract.get("minimum_cross_local_asr"), 0.06)
            and clean_mae_delta_ratio(cross_best)
            <= _as_float(thesis_contract.get("cross_clean_mae_delta_ratio_max"), 0.05)
        ),
        "cross_strong_bar_met": (
            main_local_asr(cross_best, thesis_contract) >= _as_float(thesis_contract.get("strong_cross_local_asr"), 0.07)
            and clean_mae_delta_ratio(cross_best)
            <= _as_float(thesis_contract.get("cross_clean_mae_delta_ratio_strong_max"), 0.03)
        ),
    }
