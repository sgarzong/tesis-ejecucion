"""Arma el conjunto de entrenamiento a partir de la revisión manual (tesis.sort_gui).

    python -m tesis.build_dataset [data/clasificadas/revision.csv]

- data/entrenamiento/<clase>/<foto>: enlaces duros a las copias de data/clasificadas. No ocupan espacio
  extra y quedan congelados aunque se siga revisando. Se rehace completo en cada ejecución, junto con una
  copia de la revisión usada (data/entrenamiento/revision.csv).
- data/manifest.csv: path,label,store,date,source,verified (formato de tesis.split). Solo fotos revisadas
  y no descartadas.

Ruta de origen: <fotos>/<cadena y supervisor, en cualquier orden>/<mercaderista>/<LOCAL>/.../<archivo>.
"""
import difflib
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path, PureWindowsPath

import pandas as pd

from .sort_gui import DISCARD, copy_name

FOTOS = PureWindowsPath(r"C:\Users\sebas\Downloads\proyecto-20260627T133347Z-3-001\proyecto"
                        r"\clasificacion_ales\fotos")
CHAINS = {"TIA": "TIA", "ROSADO": "ROSADO", "FAVORITA": "FAVORITA", "SANTAMARIA": "SANTAMARIA", "CORAL": "CORAL"}
NO_STORE = "SIN_LOCAL"  # fotos sin carpeta de local: un solo grupo, así no se filtran entre particiones
SINT = Path("data/sinteticas/sinteticas.csv")  # capturas inválidas sintéticas (tesis.degrade)
# Observado en el corpus: a veces la carpeta del mercaderista falta y en su lugar del local queda una de visita.
VISIT = re.compile(r"SEMANA|PARTE|^BRENDA")
TYPO = {"HIPERMAKET": "HIPERMARKET", "HIPEMARKET": "HIPERMARKET", "HIPER": "HIPERMARKET", "COMISARROATO": "COMISARIATO"}
# Variantes revisadas a mano que las reglas no unen. Revisar data/locales.csv y agregar aquí.
ALIAS = {"ROSADO HIPERMARKET TRONCAL": "ROSADO HIPERMARKET LA TRONCAL", "TIA VINCES KVDS": "TIA VINCES"}
PREFIX = {"MI", "TIA", "SMS", "CORAL"}  # 'MI COMISARIATO X' = 'COMISARIATO X', 'TIA TIA X' = 'TIA X'


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
    chain = chain_of(parts)
    i = 2 if len(parts) > 3 and VISIT.search(norm(parts[3])) else 3
    if len(parts) <= i:
        return NO_STORE
    toks = [TYPO.get(t, t) for t in norm(parts[i]).split()]
    code = toks[0] if toks[0].isdigit() else next((t for t in toks if re.fullmatch(r"\d{3}", t)), None)
    if code:
        return f"{chain} {code}"
    while len(toks) > 1 and toks[0] in PREFIX:
        toks.pop(0)
    return f"{chain} {' '.join(toks)}"


def merge_aliases(keys):
    """Une variantes de un mismo local de la misma cadena: nombre truncado ('EL DORAD' / 'EL DORADO') o
    errata (similitud >= 0.95). Devuelve dict clave -> canónica (la más frecuente)."""
    counts = keys.value_counts()
    canon, alias = [], {}
    for k in counts.index:
        name = k.split(" ", 1)[-1]
        target = None
        if k != NO_STORE and not name[:1].isdigit():
            for c in canon:
                same_chain = c.split(" ", 1)[0] == k.split(" ", 1)[0] and not c.split(" ", 1)[-1][:1].isdigit()
                short = min(len(name), len(c.split(" ", 1)[-1]))
                if same_chain and ((short >= 10 and (c.startswith(k) or k.startswith(c)))
                                   or difflib.SequenceMatcher(None, c, k).ratio() >= 0.95):
                    target = c
                    break
        alias[k] = target or k
        if not target:
            canon.append(k)
    return alias


def date_of(name):
    """AAAA-MM-DD desde el nombre (TimePhoto_AAAAMMDD, WhatsApp Image AAAA-MM-DD, epoch en ms) o ''."""
    if m := re.match(r"(1[67]\d{11})\D", name):
        d = pd.Timestamp(int(m.group(1)), unit="ms")
    elif m := re.search(r"(20[2-3]\d)-?(\d{2})-?(\d{2})", name):
        d = pd.to_datetime("".join(m.groups()), format="%Y%m%d", errors="coerce")
    else:
        return ""
    return d.strftime("%Y-%m-%d") if pd.notna(d) and 2023 <= d.year <= 2026 else ""


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
    m["store"] = m.store.map(merge_aliases(m.store))
    # Auditoría de la agrupación: qué carpetas de origen quedaron en cada local.
    (m.assign(local_carpeta=m.carpeta.str.split(" \\| ").map(lambda p: p[3] if len(p) > 3 else ""))
      .groupby(["store", "local_carpeta"]).size().rename("fotos").reset_index()
      .to_csv("data/locales.csv", index=False))
    m.drop(columns="carpeta")[["path", "label", "store", "date", "source", "verified"]].to_csv(
        "data/manifest.csv", index=False)
    print(f"{len(m)} fotos en {out} · {m.store.nunique()} locales · sin fecha {(m.date == '').sum()}"
          f" · sin local {(m.store == NO_STORE).sum()} · sin copia (omitidas) {missing}")
    print(m.groupby(["label", "source"]).agg(fotos=("path", "size"), locales=("store", "nunique")).to_string())


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:]))
