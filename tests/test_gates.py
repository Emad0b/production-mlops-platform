import pytest
from mlops_platform.registry.promote import gate, promote
from mlops_platform.config import SCHEMA_VERSION


def candidate():
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "same",
        "metrics": {"pr_auc": 0.7, "recall": 0.75, "precision": 0.6, "brier": 0.1},
    }


def test_requires_held_out_minimums_and_no_regression():
    current = candidate()
    assert gate(current)["passed"]
    weaker = candidate()
    weaker["metrics"]["pr_auc"] = 0.1
    assert not gate(weaker)["passed"]
    incompatible = candidate()
    incompatible["schema_version"] = "unknown"
    assert not gate(incompatible)["passed"]
    regressed = candidate()
    regressed["metrics"]["recall"] = 0.61
    assert not gate(regressed, current)["passed"]
    different = candidate()
    different["dataset"] = "other"
    assert not gate(different, current)["passed"]


def test_promotion_cannot_skip_explicit_approval():
    with pytest.raises(ValueError, match="Explicit approval"):
        promote("does-not-exist")


def test_nonfinite_scores_cannot_bypass_the_gate():
    invalid = candidate()
    invalid["metrics"]["pr_auc"] = float("nan")
    assert not gate(invalid)["passed"]
