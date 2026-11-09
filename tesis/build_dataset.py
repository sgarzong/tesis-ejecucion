"""Arma el conjunto de entrenamiento a partir de la revisión manual (tesis.sort_gui).

    python -m tesis.build_dataset [data/clasificadas/revision.csv]

- data/entrenamiento/<clase>/<foto>: enlaces duros a las copias de data/clasificadas. No ocupan espacio
  extra y quedan congelados aunque se siga revisando. Se rehace completo en cada ejecución, junto con una
  copia de la revisión usada (data/entrenamiento/revision.csv).
- data/manifest.csv: path,label,store,date,source,verified,declarado (formato de tesis.split). Solo fotos
  revisadas, no descartadas, con local conocido y sin cuasi-duplicados dentro del local.

Ruta de origen: <fotos>/<cadena y supervisor, en cualquier orden>/<mercaderista>/<LOCAL>/.../<archivo>.

Fotos de ODK (sección 5.6): las revisadas en tesis.sort_gui (ODK_REVIEWS) entran con source=odk, verified=1 y
el mueble declarado. Su local se lleva a la clave del histórico (TIA 149, SANTAMARIA OFELIA...), así un local de
prueba sigue en prueba; los grupos del histórico no cambian al llegar ODK.
"""
import difflib
import os
import re
import shutil
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PureWindowsPath

import pandas as pd

from .sort_gui import DISCARD, copy_name
from .split import dhash, near_duplicates

FOTOS = PureWindowsPath(r"C:\Users\sebas\Downloads\proyecto-20260627T133347Z-3-001\proyecto"
                        r"\clasificacion_ales\fotos")
CHAINS = {"TIA": "TIA", "ROSADO": "ROSADO", "FAVORITA": "FAVORITA", "SANTAMARIA": "SANTAMARIA", "CORAL": "CORAL"}
NO_STORE = "SIN_LOCAL"  # fotos sin carpeta de local: un solo grupo, así no se filtran entre particiones
SINT = Path("data/sinteticas/sinteticas.csv")  # capturas inválidas sintéticas (tesis.degrade)
# Observado en el corpus: a veces la carpeta del mercaderista falta y en su lugar del local queda una de visita.
VISIT = re.compile(r"SEMANA|PARTE|^BRENDA")
TYPO = {"HIPERMAKET": "HIPERMARKET", "HIPEMARKET": "HIPERMARKET", "HIPER": "HIPERMARKET", "COMISARROATO": "COMISARIATO"}
# Variantes revisadas a mano que las reglas no unen. Revisar data/locales.csv y agregar aquí.
ALIAS = {"ROSADO HIPERMARKET TRONCAL": "ROSADO HIPERMARKET LA TRONCAL", "TIA VINCES KVDS": "TIA VINCES",
         "ROSADO HIPERMARKET PASAJE": "ROSADO P G"}  # ODK; en el histórico, 'HIPERMARKEY PASAJE' cae en P G
PREFIX = {"MI", "TIA", "SMS", "CORAL"}  # 'MI COMISARIATO X' = 'COMISARIATO X', 'TIA TIA X' = 'TIA X'
GENERIC = {"HIPERMARKET", "COMISARIATO", "FERRISARIATO", "SUPERMAXI", "SUPER", "AKI", "MEGA", "GRAN", "CENTRO",
           "MALL", "PLAZA", "LA", "EL", "DE", "DEL", "LOS", "LAS", "Y"}
HASHES = Path("data/hashes.csv")  # caché de dhash por foto
ODK_REVIEWS = {Path("data/clasificadas_odk_muebles"): Path("data/odk/muebles_detalle.csv"),  # tesis.odk_candidatas
               Path("data/clasificadas_odk"): Path("data/odk/candidatas_detalle.csv")}
# Cadena en ODK (razón social o nombre corto) -> cadena del histórico. Explícito: un mercaderista o un
# autoservicio independiente puede llamarse Santamaría o Rosado. Liris (Del Portal) solo aparece en ODK.
ODK_CHAINS = {r"^TIA$|ASOCIADAS TIA": "TIA", r"^FAVORITA$|CORPORACION FAVORITA": "FAVORITA",
              r"^(EL )?ROSADO$|CORPORACION EL ROSADO": "ROSADO", r"MEGA SANTA ?MARIA": "SANTAMARIA",
              r"^CORAL$|ORTIZ CORAL": "CORAL", r"^LIRIS": "LIRIS"}
CODED = {"TIA", "FAVORITA"}  # cadenas cuyos locales se identifican por número


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", s).split())


def chain_of(parts):
    for p in parts[:2]:
        for key, chain in CHAINS.items():
            if key in norm(p).split():
                return chain
    return "OTRA"


def store_of(parts):
    """Clave de agrupamiento 'CADENA LOCAL'. Locales con código (TIA, Favorita) se identifican por el número:
    el sufijo con iniciales ('417 - CALDERON CSS') cambia con el supervisor."""
    i = 2 if len(parts) > 3 and VISIT.search(norm(parts[3])) else 3
    if len(parts) <= i:
        return NO_STORE
    return store_key(chain_of(parts), parts[i])


def store_key(chain, name):
    toks = [TYPO.get(t, t) for t in norm(name).split()]
    code = toks[0] if toks[0].isdigit() else next((t for t in toks if re.fullmatch(r"\d{3}", t)), None)
    if code:
        return f"{chain} {code}"
    while len(toks) > 1 and toks[0] in PREFIX:
        toks.pop(0)
    return f"{chain} {' '.join(toks)}"


def core(key):
    """Nombre del local sin la cadena ni palabras genéricas: 'ROSADO HIPERMARKET VIA DAULE' -> 'VIA DAULE'."""
    return " ".join(t for t in key.split(" ", 1)[-1].split() if t not in GENERIC)


def merge_aliases(keys, keep=frozenset()):
    """Une en un grupo los locales de nombre parecido de la misma cadena. Unir dos locales distintos solo
    agranda un grupo; dejar separados dos nombres del mismo local filtra fotos entre particiones. Por eso la
    regla es generosa: un núcleo contenido en el otro (DAULE en VIA DAULE) o similitud >= 0.85 (erratas,
    truncados), con cierre transitivo. Los locales con código no se tocan. Devuelve dict clave -> grupo.
    `keep`: grupos ya asignados (los del histórico al sumar ODK): siguen siendo raíz y no se unen entre sí,
    así su partición en stores_split.csv no cambia."""
    counts = keys.value_counts()
    parent = {k: k for k in counts.index}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    named = [k for k in counts.index if k != NO_STORE and core(k) and not k.split(" ", 1)[-1].isdigit()]
    for i, a in enumerate(named):  # en orden de frecuencia: la raíz de cada grupo es su nombre más usado
        for b in named[i + 1:]:
            if a.split(" ", 1)[0] != b.split(" ", 1)[0]:
                continue
            ca, cb = core(a), core(b)
            if set(ca.split()) <= set(cb.split()) or set(cb.split()) <= set(ca.split()) \
                    or difflib.SequenceMatcher(None, ca, cb).ratio() >= .85:
                ra, rb = find(a), find(b)
                if ra in keep and rb in keep:
                    continue
                if rb in keep:
                    ra, rb = rb, ra
                parent[rb] = ra
    return {k: find(k) for k in counts.index}


def date_of(name):
    """AAAA-MM-DD desde el nombre (TimePhoto_AAAAMMDD, WhatsApp Image AAAA-MM-DD, epoch en ms) o ''."""
    if m := re.match(r"(1[67]\d{11})\D", name):
        d = pd.Timestamp(int(m.group(1)), unit="ms")
    elif m := re.search(r"(20[2-3]\d)-?(\d{2})-?(\d{2})", name):
        d = pd.to_datetime("".join(m.groups()), format="%Y%m%d", errors="coerce")
    else:
        return ""
    return d.strftime("%Y-%m-%d") if pd.notna(d) and 2023 <= d.year <= 2026 else ""


def odk_store(cadena, local):
    """Clave 'CADENA LOCAL' de una foto de ODK, o None fuera del canal moderno."""
    c = norm(str(cadena))
    chain = next((v for k, v in ODK_CHAINS.items() if re.search(k, c)), None)
    if chain is None:
        return None
    if chain not in CODED:  # numeración propia de ODK ('5.- OFELIA'); el histórico usa el nombre
        local = re.sub(r"^\s*\d+\s*\.?\s*-?\s*", "", str(local))
    return store_key(chain, local)


def odk_rows():
    """Fotos de ODK revisadas, con la clave de local sin unir todavía a los grupos del histórico."""
    rows = []
    for src, detail in ODK_REVIEWS.items():
        if not (src / "revision.csv").exists():
            continue
        r = pd.read_csv(src / "revision.csv", encoding="utf-8-sig").drop_duplicates("path", keep="last")
        d = r[r.label != DISCARD].merge(pd.read_csv(detail).drop(columns="label"), on="path")
        for x in d.to_dict("records"):
            store, copy = odk_store(x["cadena"], x["local"]), src / x["label"] / copy_name(x["path"])
            if store and copy.exists():
                rows.append({"path": copy.as_posix(), "label": x["label"], "store": store,
                             "date": str(x["fecha"])[:10], "source": "odk", "verified": 1,
                             "declarado": x.get("declarada", ""),
                             "carpeta": f"ODK | {x['form']} | {x['cadena']} | {x['local']}"})
    return pd.DataFrame(rows)


def safe_dhash(path):
    try:
        return dhash(path)
    except OSError:
        return 0  # ilegible: todas las ilegibles de un local cuentan como una sola


def dhashes(paths):
    """dhash de cada foto, con caché por nombre de archivo (estable entre reconstrucciones)."""
    cache = pd.read_csv(HASHES, dtype=str).set_index("name").hash.to_dict() if HASHES.exists() else {}
    names = [p.rsplit("/", 1)[-1] for p in paths]
    todo = [p for p, n in zip(paths, names) if n not in cache]
    with ProcessPoolExecutor() as ex:
        for p, h in zip(todo, ex.map(safe_dhash, todo, chunksize=64)):
            cache[p.rsplit("/", 1)[-1]] = str(h)
    pd.DataFrame({"name": list(cache), "hash": list(cache.values())}).to_csv(HASHES, index=False)
    return [int(cache[n]) for n in names]


def main(revision=Path("data/clasificadas/revision.csv")):
    out, src = Path("data/entrenamiento"), revision.parent
    r = pd.read_csv(revision, encoding="utf-8-sig").drop_duplicates("path", keep="last")
    r = r[r.label != DISCARD]
    parts = r.path.map(lambda p: PureWindowsPath(p).relative_to(FOTOS).parts[:-1])
    if out.exists():
        shutil.rmtree(out)  # solo contiene enlaces: las copias de data/clasificadas no se tocan
    rows, missing = [], 0
    for path, label, pp in zip(r.path, r.label, parts):
        name = copy_name(path)
        if not (src / label / name).exists():
            missing += 1
            continue
        (out / label).mkdir(parents=True, exist_ok=True)
        os.link(src / label / name, out / label / name)
        rows.append({"path": (out / label / name).as_posix(), "label": label, "store": store_of(pp),
                     "date": date_of(PureWindowsPath(path).name), "source": "hist", "verified": 1,
                     "carpeta": " | ".join(pp)})
    shutil.copy2(revision, out / "revision.csv")
    m = pd.DataFrame(rows)
    if SINT.exists():  # tesis.degrade: heredan local y fecha de su origen (misma partición, sin fuga)
        s = pd.read_csv(SINT)
        src_of = m.assign(origen=m.path.str.rsplit("/", n=1).str[-1]).set_index("origen")
        s = s[s.origen.isin(src_of.index)]  # si el origen se descartó después, su sintética sale
        m = pd.concat([m, src_of.loc[s.origen, ["store", "date", "carpeta"]].reset_index(drop=True).assign(
            path=s.path.values, label="captura_invalida", source="sint", verified=0)], ignore_index=True)
    m["store"] = m.store.replace(ALIAS)
    groups = merge_aliases(m.store)
    m["store"] = m.store.map(groups)
    o = odk_rows()
    if len(o):  # después del histórico: sus grupos quedan fijos y ODK se une a ellos
        o["store"] = o.store.replace(ALIAS)
        seen = o.store.isin(groups.keys())  # nombre que el histórico ya tenía: su grupo, sin volver a comparar
        o.loc[seen, "store"] = o.store[seen].map(groups)
        o.loc[~seen, "store"] = o.store[~seen].map(merge_aliases(pd.concat([m.store, o.store[~seen]]),
                                                                  keep=set(m.store)))
        m = pd.concat([m, o], ignore_index=True)
    # Sin local conocido no se puede garantizar que la foto no sea de un local de prueba: fuera.
    gone = m.store == NO_STORE
    # Cuasi-duplicados (ráfagas del mismo mueble) dentro de cada local: queda la primera por fecha.
    h = m[(m.source != "sint") & ~gone].sort_values(["store", "date", "path"])
    h["hash"] = pd.Series(dhashes(h.path), index=h.index, dtype=object)  # 64 bits: no cabe en int64
    drop = m.index[gone].union(h.index[near_duplicates(h)])
    for p in m.path[drop]:
        if p.startswith(out.as_posix()):  # enlaces propios; las sintéticas de data/sinteticas no se borran
            Path(p).unlink()
    print(f"fuera: sin local {gone.sum()} · cuasi-duplicados {len(drop) - gone.sum()}")
    m = m.drop(drop)
    # Auditoría de la agrupación: qué carpetas de origen quedaron en cada local.
    (m.assign(local_carpeta=m.carpeta.str.split(" \\| ").map(lambda p: p[3] if len(p) > 3 else ""))
      .groupby(["store", "local_carpeta"]).size().rename("fotos").reset_index()
      .to_csv("data/locales.csv", index=False))
    m.reindex(columns=["path", "label", "store", "date", "source", "verified", "declarado"]).to_csv(
        "data/manifest.csv", index=False)
    sizes = m.store.value_counts()
    print(f"{len(m)} fotos en {out} · {len(sizes)} locales (el mayor: {sizes.index[0]}, {sizes.iloc[0]} fotos)"
          f" · sin fecha {(m.date == '').sum()} · sin copia (omitidas) {missing}")
    print(m.groupby(["label", "source"]).agg(fotos=("path", "size"), locales=("store", "nunique")).to_string())


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
