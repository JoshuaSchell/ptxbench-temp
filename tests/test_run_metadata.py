from ptxbench.run_metadata import (
    default_paper_protocol,
    normalize_problem_ids,
    protocol_differences,
    protocol_signature,
    sha256_text,
)


def test_sha256_text_is_stable() -> None:
    assert sha256_text("ptxbench") == sha256_text("ptxbench")
    assert sha256_text("ptxbench") != sha256_text("kernelbench")


def test_normalize_problem_ids_sorts_and_deduplicates() -> None:
    assert normalize_problem_ids([3, 1, 3, 2]) == [1, 2, 3]
    assert normalize_problem_ids(None) is None


def test_default_paper_protocol_has_level1_defaults() -> None:
    protocol = default_paper_protocol().to_dict()
    assert protocol["level"] == 1
    assert protocol["precision"] == "fp32"
    assert protocol["one_shot"] is True


def test_default_paper_protocol_tracks_level_specific_pilot_ids() -> None:
    protocol = default_paper_protocol(level=4).to_dict()
    assert protocol["level"] == 4
    assert protocol["pilot_problem_ids"] == (1, 5, 10, 15, 20)


def test_default_paper_protocol_agentic_defaults() -> None:
    protocol = default_paper_protocol(level=1, track="agentic").to_dict()
    assert protocol["track"] == "agentic"
    assert protocol["one_shot"] is False
    assert protocol["max_steps"] == 5
    assert protocol["max_tool_calls"] == 4
    assert protocol["official_eval_seed"] == 42
    assert protocol["dev_eval_seed"] == 7
    assert protocol["dev_eval_correct_trials"] == 2
    assert protocol["dev_eval_profile_enabled"] is False
    assert protocol["dev_eval_profile_tool"] == "ncu"
    assert protocol["dev_eval_profile_trials"] == 1


def test_protocol_signature_and_differences_ignore_non_contract_fields() -> None:
    baseline = default_paper_protocol(level=1, track="agentic").to_dict()
    observed = {
        **baseline,
        "extra_field": "ignore-me",
    }
    assert protocol_signature(observed) == protocol_signature(baseline)

    drifted = {
        **baseline,
        "official_eval_seed": 99,
    }
    assert protocol_differences(baseline, drifted) == {"official_eval_seed": (42, 99)}
