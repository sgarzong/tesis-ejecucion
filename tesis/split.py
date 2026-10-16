"""Partición agrupada por local (sección 5.4) y deduplicación perceptual.

Manifiesto de entrada (CSV), una fila por imagen:
    path, label, store, date (AAAA-MM-DD), source (hist|odk), verified (0|1)
`store` es el nombre normalizado del local: es la clave de agrupamiento.

Uso:
    python -m tesis.split data/manifest.csv data/stores_split.csv
Si stores_split.csv ya existe, los locales asignados conservan su partición
(consistencia entre corpus: un local de prueba sigue en prueba cuando llega ODK).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import MAX_PER_STORE, SEED, TEMPORAL_CUTOFF


def assign_stores(df, prev=None, test=0.2, val=0.1):
    """Devuelve DataFrame store -> split (train/val/test), estratificado por clase dentro del grupo."""
    prev = prev if prev is not None else pd.DataFrame(columns=["store", "split"])
    new = df[~df.store.isin(prev.store)]
    if new.empty:
        return prev
    names = new.store.unique()
    if len(names) < 2 * round((1 - test) / val):
        # Pocos locales nuevos (goteo de ODK): no alcanzan para estratificar, se sortean.
        rng = np.random.default_rng(SEED)
        split = rng.choice(["train", "val", "test"], len(names), p=[1 - test - val, val, test])
        return pd.concat([prev, pd.DataFrame({"store": names, "split": split})], ignore_index=True)
    split = pd.Series("train", index=new.index)
    _, te = next(StratifiedGroupKFold(round(1 / test), shuffle=True, random_state=SEED)
                 .split(new, new.label, new.store))
    split.iloc[te] = "test"
    rest = new[split == "train"]
    _, va = next(StratifiedGroupKFold(round((1 - test) / val), shuffle=True, random_state=SEED)
                 .split(rest, rest.label, rest.store))
    split[rest.index[va]] = "val"
    assigned = pd.DataFrame({"store": new.store, "split": split}).drop_duplicates("store")
    return pd.concat([prev, assigned], ignore_index=True)


def experiment(df, stores, exp="A"):
    """Devuelve dict train/val/test de DataFrames para el experimento A (locales) o B (temporal)."""
    df = df.merge(stores, on="store")
    if exp == "B":
        hist = df[df.source == "hist"]
        before, after = hist[hist.date < TEMPORAL_CUTOFF], hist[hist.date >= TEMPORAL_CUTOFF]
        parts = {"train": before[before.split == "train"], "val": before[before.split == "val"],
                 "test": after[after.split == "test"]}
    else:
        parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    # Prueba solo con etiquetas verificadas a mano (sección 5.3).
    parts["test"] = parts["test"][parts["test"].verified == 1]
    # ponytail: muestreo uniforme por local; estratificar por visita si las ráfagas dominan.
    parts["train"] = parts["train"].sample(frac=1, random_state=SEED).groupby("store").head(MAX_PER_STORE)
    return parts


def dhash(path, size=8):
    """Huella perceptual de 64 bits (difference hash)."""
    px = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (px[:, 1:] > px[:, :-1]).flatten()), 2)


def near_duplicates(df, max_dist=4):
    """Marca como duplicada cada imagen a distancia de Hamming <= max_dist de una anterior del mismo local.

    Requiere columna `hash` (dhash). Compara solo dentro del local: O(n²) por local, ~1 200 imágenes máx.
    """
    dup = pd.Series(False, index=df.index)
    for _, g in df.groupby("store"):
        kept = []
        for i, h in g.hash.items():
            if any(bin(h ^ k).count("1") <= max_dist for k in kept):
                dup[i] = True
            else:
                kept.append(h)
    return dup


if __name__ == "__main__":
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.read_csv(manifest)
    prev = pd.read_csv(out) if out.exists() else None
    stores = assign_stores(df[df.label.notna()], prev)
    stores.to_csv(out, index=False)
    print(df.merge(stores, on="store").groupby(["split", "label"]).size().unstack(fill_value=0))
