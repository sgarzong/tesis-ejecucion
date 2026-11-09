"""Chequeo mínimo de la partición y las métricas. Ejecutar: python -m pytest tests  (o python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tesis.build_dataset import NO_STORE, date_of, merge_aliases, odk_store, store_of
from tesis.config import CLASSES
from tesis.degrade import degrade, shake_psf
from tesis.metrics import IDX, conditional_pred, failed_criteria, verification
from tesis.metrics import contracts as contracts_from
from tesis.sort_gui import DISCARD, copy_name, paginate, place
from tesis.split import assign_stores, cross_store_duplicates, experiment


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

    # La misma foto en un local de prueba y en uno de entrenamiento: sale la de prueba. Negras y hash 0 no cuentan.
    d = pd.DataFrame({"store": ["t", "e", "e2", "t", "e", "t", "e"], "hash": [7, 7, 7, 0, 0, 5, 5],
                      "label": ["lateral"] * 5 + ["captura_invalida"] * 2})
    assert list(cross_store_duplicates(d, ["t"])) == [True, False, True, False, False, False, False]


def test_conditional_and_verification():
    contracts = {"a": {"chiclera", "lateral"}}
    probs = np.zeros((2, len(CLASSES)))
    probs[0, IDX["cabecera"]], probs[0, IDX["chiclera"]] = 0.9, 0.1  # cabecera no está en convenio
    probs[1, IDX["lateral"]] = 1.0
    y = np.array([IDX["chiclera"], IDX["lateral"]])
    assert list(conditional_pred(probs, ["a", "a"], contracts)) == list(y)
    v = verification(probs, y, ["a", "a"], contracts)
    assert v == {"detection_rate": 1.0, "false_alarm": 0.0, "n_mismatch": 2, "n_correct": 2}
    # Convenio aproximado: muebles vistos y declarados; los rechazos y lo no verificado no cuentan.
    m = pd.DataFrame({"store": ["a", "a", "a", "b"], "label": ["chiclera", "contenido_ajeno", "lateral", "cabecera"],
                      "declarado": [None, "check_out", None, None], "verified": [1, 1, 0, 1]})
    assert contracts_from(m) == {"a": {"chiclera", "check_out"}, "b": {"cabecera"}}


def test_failed_criteria():
    ok = {"f1_macro": .9, "recall_macro": .92, "reject_recall": .95, "false_alarm": .03, "quant_drop": .01,
          "lat_p50_ms": 800, "lat_p95_ms": 1200, "mem_mb": 120, "size_mb": 5}
    assert failed_criteria(ok) == []
    assert failed_criteria({**ok, "size_mb": 25, "f1_macro": .8}) == ["f1_macro", "size_mb"]


def test_place_moves_copy(tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"x")
    out, p = tmp_path / "out", str(src)
    copies = lambda: sorted(f.parent.name for f in out.rglob("*.jpg"))
    place(out, p, None, "lateral")
    assert copies() == ["lateral"]
    place(out, p, "lateral", "cabecera")  # corrección: la copia se mueve, no se duplica
    assert copies() == ["cabecera"]
    place(out, p, "cabecera", DISCARD)
    assert copies() == [] and src.exists()
    assert copy_name(p) != copy_name(str(tmp_path / "otro" / "a.jpg"))


def test_paginate_one_class_per_page():
    rows = [{"label": c} for c in ["cabecera"] * 5 + ["lateral"] * 2]
    assert [[r["label"] for r in pg] for pg in paginate(rows, 3)] == [
        ["cabecera"] * 3, ["cabecera"] * 2, ["lateral"] * 2]


def test_store_and_date():
    tia = "TIENDAS INDUSTRIALES ASOCIADAS TIA S.A"
    # El código identifica al local aunque cambien las iniciales del supervisor o el orden de carpetas.
    assert store_of((tia, "SARA", "MERC", "417 - CALDERON CSS")) == "TIA 417"
    assert store_of(("BEATRIZ CHOEZ", tia, "MERC", "417.- CALDERON JGC", "Semana 3")) == "TIA 417"
    assert store_of(("GERARDO ORTIZ CORAL", "K", "M", "Coral batan")) == store_of(("GERALDO ORTIZ CORAL", "K", "M", "BATÁN"))
    assert store_of((tia, "CARLA CASTRO")) == NO_STORE
    assert store_of((tia, "S", "M", "TIA PRADERA 597")) == "TIA 597"
    assert store_of(("CORPORACION ROSADO", "S", "HIPERMARKET AMBATO", "SEMANA 2")) == "ROSADO HIPERMARKET AMBATO"
    assert store_of(("CORPORACION ROSADO", "S", "M", "MI COMISARIATO YAGUACHI")) == "ROSADO COMISARIATO YAGUACHI"
    keys = pd.Series(["ROSADO HIPERMARKET RIOCENTRO EL DORAD"] * 3 + ["ROSADO HIPERMARKET RIOCENTRO EL DORADO",
                     "ROSADO COMISARIATO JUAN TANGA", "ROSADO COMISARIATO JUAN TANCA", "ROSADO COMISARIATO SUR",
                     "ROSADO HIPERMARKET DAULE", "ROSADO HIPERMARKET 12 5 VIA DAULE", "ROSADO HIPERMARKET MANTA",
                     "SANTAMARIA CENTRO", "SANTAMARIA LATACUNGA CENTRO", "SANTAMARIA LATACUNGA", "TIA 417"])
    g = merge_aliases(keys)
    assert g["ROSADO HIPERMARKET DAULE"] == g["ROSADO HIPERMARKET 12 5 VIA DAULE"]  # núcleo contenido
    assert g["SANTAMARIA LATACUNGA"] == g["SANTAMARIA LATACUNGA CENTRO"]
    assert g["SANTAMARIA CENTRO"] == "SANTAMARIA CENTRO"  # solo palabras genéricas: no arrastra a nadie
    assert keys.map(g).nunique() == 8
    # ODK: misma clave que el histórico; fuera lo que no es de una cadena del canal moderno.
    assert odk_store("TIENDAS INDUSTRIALES ASOCIADAS TIA S.A.", "149 - DAULE") == "TIA 149"
    assert odk_store("Corporacion Favorita S A", "714-MEGAMAXI EL CONDADO") == "FAVORITA 714"
    assert odk_store("Mega Santamaría", "5.- OFELIA") == "SANTAMARIA OFELIA"
    assert odk_store("CORPORACION EL ROSADO S.A.", "HIPEMARKET BUIJO") == "ROSADO HIPERMARKET BUIJO"
    assert odk_store("ACOSTA SANTAMARIA EDGAR PATRICIO", "ACOSTA SANTAMARIA EDGAR PATRICIO") is None
    assert odk_store("FARCOMED/FYBECA/GPF", "FYBECA ANTONIO ANTE") is None
    # Al sumar ODK, los grupos ya asignados quedan como raíz y no se unen entre sí a través de un nombre nuevo.
    hist = {"SANTAMARIA LATACUNGA", "SANTAMARIA LATACUNGA TERMINAL"}
    g = merge_aliases(pd.Series(["SANTAMARIA LATACUNGA TERMINAL CENTRO"] * 5 + sorted(hist)), keep=hist)
    assert all(g[k] == k for k in hist) and g["SANTAMARIA LATACUNGA TERMINAL CENTRO"] in hist
    assert date_of("TimePhoto_20240410_091922.jpg") == "2024-04-10"
    assert date_of("WhatsApp Image 2025-03-02 at 10.11.12 (1).jpeg") == "2025-03-02"
    assert date_of("1712745562000.jpg") == "2024-04-10"
    assert date_of("TC_12345.jpeg") == ""


def test_degrade_keeps_overlay():
    rng = np.random.default_rng(0)
    k = shake_psf(40, rng)
    assert abs(k.sum() - 1) < 1e-9 and 38 <= np.ptp(np.nonzero(k), axis=1).max() <= 42
    img = np.full((400, 300, 3), 120, np.uint8)
    img[5:45, 240:280] = 255
    img[15:35, 250:270] = 40  # logo de Timestamp Camera: cuadro blanco con lente oscura
    out, _ = degrade(img, "oscura_movida", rng)
    assert (out[5:45, 240:280] == img[5:45, 240:280]).all()  # se estampa después de la captura: nítido
    assert out[100:300, :200].mean() < 30  # la escena queda fuera del rango de los aumentos de brillo


if __name__ == "__main__":
    test_degrade_keeps_overlay()
    test_store_and_date()
    import tempfile
    test_split_no_leak_and_stable()
    test_conditional_and_verification()
    test_failed_criteria()
    with tempfile.TemporaryDirectory() as d:
        test_place_moves_copy(Path(d))
    test_paginate_one_class_per_page()
    print("ok")
