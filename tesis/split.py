"""Partición agrupada por local (sección 5.4) y deduplicación perceptual.

Manifiesto de entrada (CSV), una fila por imagen:
    path, label, store, date (AAAA-MM-DD), source (hist|odk), verified (0|1)
`store` es el nombre normalizado del local: es la clave de agrupamiento.

Uso, después de tesis.build_dataset (que ya deduplicó):
    python -m tesis.split data/manifest.csv data/stores_split.csv [--rehacer]
Si stores_split.csv ya existe, los locales asignados conservan su partición
(consistencia entre corpus: un local de prueba sigue en prueba cuando llega ODK).
Escribe además data/particion.csv, la lista congelada de fotos de cada partición y experimento (freeze).
"""
import hashlib
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
    # Validación y prueba solo con etiquetas verificadas a mano (sección 5.3): sin sintéticas ni etiquetas automáticas.
    for s in ("val", "test"):
        parts[s] = parts[s][parts[s].verified == 1]
    # ponytail: muestreo uniforme por local; estratificar por visita si las ráfagas dominan.
    parts["train"] = parts["train"].sample(frac=1, random_state=SEED).groupby("store").head(MAX_PER_STORE)
    return parts


def dhash(path, size=8):
    """Huella perceptual de 64 bits (difference hash)."""
    img = Image.open(path)
    img.draft("L", (size * 8, size * 8))  # JPEG: decodifica ya reducida, decenas de veces más rápido
    px = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
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


def cross_store_duplicates(df):
    """Marca todas las copias de una foto archivada en más de un local (TIA 133 y TIA 546 comparten 16): no se sabe
    de cuál de los dos es, y con cualquiera que quede un local tendría fotos de otro. No mira la partición, así que
    se deduplica antes de sortearla. Requiere `hash` y que ya no haya cuasi-duplicados dentro de cada local. Solo
    hash idéntico: entre locales distintos, a distancia 1–4 coinciden fotos diferentes (la etiqueta concuerda en
    menos de dos tercios de los pares, y ninguno comparte nombre de archivo). Las capturas inválidas no se comparan:
    oscuras o planas, la huella queda en el logo y la superposición, y fotos negras de días distintos coinciden
    (freeze comprueba que no sean el mismo archivo). Hash 0: ilegible o plana."""
    d = df[(df.hash != 0) & (df.label != "captura_invalida")]
    return d.duplicated("hash", keep=False).reindex(df.index, fill_value=False)


def freeze(df, stores, out, hashes=None, force=False):
    """Congela la partición en `out`: una fila por foto con su local, su partición y su papel en los experimentos A
    y B (train, val, test o vacío). tesis.train lee esta lista, no la vuelve a derivar. Comprueba que ningún local
    ni ninguna foto esté en dos particiones y, con `hashes` (nombre de archivo -> dhash), que tampoco haya dos
    fotos con la misma huella a cada lado, salvo capturas inválidas que sean archivos distintos. Si ya hay una
    lista con otra validación u otra prueba, falla: no se tocan después de ver resultados (`force` la rehace)."""
    p = df.merge(stores, on="store")[["path", "label", "store", "date", "source", "verified", "split"]]
    for exp in "AB":
        role = {path: s for s, part in experiment(df, stores, exp).items() for path in part.path}
        p[exp] = p.path.map(role).fillna("")
    assert p.path.is_unique and stores.store.is_unique, "una foto o un local aparece dos veces"
    if hashes is not None:
        h = p.assign(hash=p.path.str.rsplit("/", n=1).str[-1].map(hashes)).dropna(subset="hash")
        for _, g in h[(h.hash != "0") & h.groupby("hash").split.transform("nunique").gt(1)].groupby("hash"):
            assert (g.label == "captura_invalida").all(), f"misma foto en dos particiones: {g.path.tolist()}"
            files = [Path(x).read_bytes() for x in g.path]
            assert len(set(files)) == len(files), f"mismo archivo en dos particiones: {g.path.tolist()}"
    if out.exists() and not force:
        old = pd.read_csv(out, keep_default_na=False)
        for exp in "AB":
            for s in ("val", "test"):
                assert set(old.path[old[exp] == s]) == set(p.path[p[exp] == s]), (
                    f"{out} ya está congelada y cambiaría {s} del experimento {exp}; --rehacer la vuelve a escribir "
                    "(solo antes de entrenar: después, la validación y la prueba no se tocan)")
    p.to_csv(out, index=False)
    return p


def fingerprint(p):
    """sha256 de la partición congelada, igual con las rutas originales que con las de la copia reducida de Colab:
    nombre de la foto sin carpeta ni extensión, clase, local, partición y papel en A y B."""
    t = p.assign(foto=p.path.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0])
    cols = t[["foto", "label", "store", "split", "A", "B"]].sort_values("foto", kind="stable")
    return hashlib.sha256(cols.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


# Locales donde el autor tomó a propósito capturas inválidas para evaluar (2026-11-16): son de prueba por diseño.
# Los mercaderistas casi no producen esas fotos, y sorteadas a entrenamiento no habría con qué medir su recall.
# La superposición solo nombra Coral de la Y y Supermaxi Iñaquito (Favorita 103); el autor indicó que también tomó
# fotos en Santa María Iñaquito, así que los tres van a prueba y ninguna de esas fotos puede ser de un local visto.
PRUEBA_FIJA = ("CORAL DE LA Y", "FAVORITA 103", "SANTAMARIA INAQUITO")

if __name__ == "__main__":
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = pd.read_csv(manifest)
    df = df[df.label.notna()]
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame({"store": PRUEBA_FIJA, "split": "test"})
    # Las capturas inválidas reales se estratifican aparte de las sintéticas: son pocas y hay que repartirlas.
    stores = assign_stores(df.assign(label=df.label.where(df.source != "sint", "sintetica")), prev)
    stores.to_csv(out, index=False)
    hashes = Path("data/hashes.csv")
    frozen = out.with_name("particion.csv")
    p = freeze(df, stores, frozen, pd.read_csv(hashes, dtype=str).set_index("name").hash if hashes.exists() else None,
               force="--rehacer" in sys.argv)
    print(p.groupby(["split", "label"]).size().unstack(fill_value=0))
    for exp in "AB":
        print(f"experimento {exp}:", p[p[exp] != ""].groupby(exp).size().to_dict())
    print(f"{frozen} congelada, huella {fingerprint(p)}")
