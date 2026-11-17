"""Convenio de cada local desde fuentes que no dependen de las etiquetas de las fotos (secciones 5.6 y 5.9).

    python -m tesis.convenios        # data/convenios.csv + data/convenios_emparejamiento.csv

Fuentes, en data/convenios/ (no se versionan):
- maestra_moderno_jun25.xlsx y maestra_in_and_out_jun25.xlsx: maestras de muebles de Industrias Ales de junio de
  2025. Cubren Tía, Santa María, Coral y El Rosado; Favorita no está.
- catalogo_muebles_20261116.xlsx: la maestra vigente, de la que sale la lista con que el formulario de ODK despliega
  los muebles de cada local.

Son maestras de muebles alquilados: no traen los ganados en negociación (acuerdos comerciales, extravisibilidad),
así que un local puede tener en sus fotos muebles que su convenio no lista. La clase sale del nombre del mueble en
el convenio (MUEBLES CONVENIO, `mueble`) y, si ese nombre no está en la taxonomía, del tipo (MUEBLES, `tipo`).

Los convenios cambian con el tiempo y cada fuente es una foto de un día, así que cada una rige solo las fotos
tomadas a menos de config.VENTANA_CONVENIO días de su fecha (tesis.metrics.contract_keys). Las demás fotos no entran a la
verificación contra convenio: no hay con qué saber qué tenía contratado el local ese día.

El local se empareja por nombre con los del manifiesto, sin mirar etiquetas ni particiones: la clave de
tesis.build_dataset.odk_store; si no existe, el local al que build_dataset llevó ese mismo nombre de ODK
(data/locales.csv); si no, la regla de merge_aliases contra los locales de la misma cadena, cuando da un solo
candidato. data/convenios_emparejamiento.csv deja cada fila de las fuentes con el local que recibió y por qué.
"""
import difflib
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

from .build_dataset import ALIAS, core, norm, odk_store
from .config import CLASE_DECLARADA

DIR = Path("data/convenios")
MAESTRA, CATALOGO = "maestra_2025-06", "catalogo_2026-11"
FECHA = {MAESTRA: "2025-06-30", CATALOGO: "2026-11-16"}
M ="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def xlsx(path):
    """Primera hoja de un .xlsx como DataFrame de texto. Es un zip de XML: no hace falta openpyxl."""
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(M + "t"))
          for si in ET.parse(z.open("xl/sharedStrings.xml")).getroot().findall(M + "si")
          ] if "xl/sharedStrings.xml" in z.namelist() else []  # hay libros que guardan el texto en la celda
    rels = {r.get("Id"): r.get("Target") for r in ET.parse(z.open("xl/_rels/workbook.xml.rels")).getroot()}
    target = rels[ET.parse(z.open("xl/workbook.xml")).getroot().find(M + "sheets")[0].get(R + "id")].lstrip("/")
    rows = []
    for row in ET.parse(z.open(target if target.startswith("xl/") else "xl/" + target)).getroot().iter(M + "row"):
        vals = {}
        for c in row.findall(M + "c"):
            v, n = c.find(M + "v"), 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                n = n * 26 + ord(ch) - 64
            if c.get("t") == "s":
                vals[n - 1] = ss[int(v.text)]
            elif c.get("t") == "inlineStr":
                vals[n - 1] = "".join(t.text or "" for t in c.iter(M + "t"))
            elif v is not None:
                vals[n - 1] = v.text
        rows.append(vals)
    head = {i: " ".join(str(h).split()).upper() for i, h in rows[0].items()}
    return pd.DataFrame([{head[i]: " ".join(str(v).split()) for i, v in r.items() if i in head} for r in rows[1:] if r]
                        ).fillna("")


def fuentes(d=DIR):
    """Una fila por (fuente, cadena, local, tipo de mueble), tal como viene, sin emparejar."""
    out = []
    for f in sorted(d.glob("*.xlsx")):
        m = xlsx(f)
        cols = (("CADENA", "LOCAL", "LOCAL_ID", "MUEBLE", "TIPO", "NEGOCIO") if "LOCAL_ID" in m else
                ("CADENA", "LOCAL", "COD LOCAL/ALES", "MUEBLES CONVENIO", "MUEBLES", "NEGOCIO"))
        out.append(m[list(cols)].set_axis(["cadena", "local", "codigo", "mueble", "tipo", "negocio"], axis=1).assign(
            fuente=CATALOGO if "LOCAL_ID" in m else MAESTRA, archivo=f.name))
    f = pd.concat(out, ignore_index=True).fillna("")
    f["mueble"], f["tipo"] = f.mueble.str.upper(), f.tipo.str.upper()
    # La maestra de 2025 no usa el nombre CHICLERA: en Tía, el check out de Mars (CHECK OUT SIMPLE o CHECK OUT
    # LATERAL DOBLE) es lo que el catálogo de 2026 separa en CHECK OUT y CHICLERA. De los 119 locales con esa fila
    # que siguen en el catálogo, 117 tienen CHICLERA; de los 48 de Santa María y Coral con CHECK OUT de Mars,
    # ninguno. La regla sale de comparar las dos maestras, no de las etiquetas.
    tia = f[(f.fuente == MAESTRA) & (f.negocio == "MARS") & f.tipo.isin(["CHECK OUT SIMPLE", "CHECK OUT LATERAL DOBLE"])]
    return pd.concat([f, tia.assign(mueble="CHICLERA", tipo="CHICLERA (regla de Tía)")], ignore_index=True)


def emparejar(cadena, local, stores, por_nombre):
    """(local del manifiesto o None, cómo se emparejó)."""
    key = odk_store(cadena, local)
    if not key:
        return None, "fuera del canal moderno"
    key = ALIAS.get(key, key)
    chain, name = key.split(" ", 1)
    if key in stores:
        return key, "clave"
    if (chain, norm(local)) in por_nombre:
        return por_nombre[chain, norm(local)], "nombre de ODK en locales.csv"
    if not name.isdigit() and core(key):
        ck = set(core(key).split())
        cand = [s for s in stores if s.split(" ", 1)[0] == chain and not s.split(" ", 1)[-1].isdigit() and core(s)
                and (ck <= set(core(s).split()) or set(core(s).split()) <= ck
                     or difflib.SequenceMatcher(None, core(key), core(s)).ratio() >= .85)]
        if len(cand) == 1:
            return cand[0], "difuso"
    return None, "sin local en el manifiesto"


def main():
    f = fuentes()
    stores = set(pd.read_csv("data/manifest.csv", usecols=["store"]).store)
    loc = pd.read_csv("data/locales.csv", dtype=str).fillna("")
    por_nombre = {(s.split(" ", 1)[0], norm(c)): s for s, c in zip(loc.store, loc.local_carpeta)}
    par = {k: emparejar(*k, stores, por_nombre) for k in set(zip(f.cadena, f.local))}
    f["store"] = [par[k][0] for k in zip(f.cadena, f.local)]
    f["como"] = [par[k][1] for k in zip(f.cadena, f.local)]
    f["clase"] = f.mueble.map(CLASE_DECLARADA).fillna(f.tipo.map(CLASE_DECLARADA)).fillna("")
    f["fecha"] = f.fuente.map(FECHA)
    f.to_csv("data/convenios_emparejamiento.csv", index=False)
    c = f[f.store.notna() & (f.clase != "")]
    c = c[["store", "clase", "fuente", "fecha"]].drop_duplicates().sort_values(["store", "fuente", "clase"])
    c.to_csv("data/convenios.csv", index=False)
    moderno = f[f.como != "fuera del canal moderno"]
    print(moderno.drop_duplicates(["fuente", "cadena", "local"]).groupby(["fuente", "como"]).size().to_string())
    print("fuera de la taxonomía:", moderno[moderno.clase == ""].groupby("fuente").mueble.value_counts().to_dict())
    print(c.groupby("fuente").agg(locales=("store", "nunique"), convenios=("clase", "size")).to_string())
    print(c.groupby(["fuente", "clase"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
