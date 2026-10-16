"""Chequeo mínimo de la partición y las métricas. Ejecutar: python -m pytest tests  (o python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tesis.config import CLASSES
from tesis.metrics import IDX, conditional_pred, failed_criteria, verification
from tesis.split import assign_stores, experiment


def test_split_no_leak_and_stable():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"store": [f"s{i // 10}" for i in range(2000)],
                       "label": rng.choice(CLASSES, 2000)})
    stores = assign_stores(df)
    assert stores.store.is_unique  # cada local en una sola partición
    assert set(stores.split) == {"train", "val", "test"}
    # Locales nuevos (ODK) no reasignan los existentes.
    more = pd.concat([df, pd.DataFrame({"store": ["nuevo"] * 10, "label": ["lateral"] * 10})])
    again = assign_stores(more, stores)
    assert again.merge(stores, on="store").eval("split_x == split_y").all()
    assert "nuevo" in set(again.store)

    df["date"] = rng.choice(["2024-06-01", "2025-11-01"], len(df))
    df["source"], df["verified"] = "hist", rng.integers(0, 2, len(df))
    for exp in "AB":
        p = experiment(df, stores, exp)
        assert set(p["train"].store).isdisjoint(p["test"].store)
        assert (p["test"].verified == 1).all()
    assert (p["test"].date >= "2025-10-01").all() and (p["train"].date < "2025-10-01").all()


def test_conditional_and_verification():
    contracts = {"a": {"chiclera", "lateral"}}
    probs = np.zeros((2, len(CLASSES)))
    probs[0, IDX["cabecera"]], probs[0, IDX["chiclera"]] = 0.9, 0.1  # cabecera no está en convenio
    probs[1, IDX["lateral"]] = 1.0
    y = np.array([IDX["chiclera"], IDX["lateral"]])
    assert list(conditional_pred(probs, ["a", "a"], contracts)) == list(y)
    v = verification(probs, y, ["a", "a"], contracts)
    assert v == {"detection_rate": 1.0, "false_alarm": 0.0, "n_mismatch": 2, "n_correct": 2}


def test_failed_criteria():
    ok = {"f1_macro": .9, "recall_macro": .92, "reject_recall": .95, "false_alarm": .03, "quant_drop": .01,
          "lat_p50_ms": 800, "lat_p95_ms": 1200, "mem_mb": 120, "size_mb": 5}
    assert failed_criteria(ok) == []
    assert failed_criteria({**ok, "size_mb": 25, "f1_macro": .8}) == ["f1_macro", "size_mb"]


if __name__ == "__main__":
    test_split_no_leak_and_stable()
    test_conditional_and_verification()
    test_failed_criteria()
    print("ok")
