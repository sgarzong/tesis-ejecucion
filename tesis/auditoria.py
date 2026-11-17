"""Auditoría ciega de las etiquetas por una segunda persona (sección 5.3: control de calidad del etiquetado).

    python -m tesis.auditoria muestra                                         # 1. sortea la muestra
    python -m tesis.sort_gui data/auditoria/cola.csv data/auditoria/segunda   # 2. la segunda persona clasifica
    python -m tesis.auditoria                                                 # 3. concordancia y desacuerdos

La muestra sale de la validación y la prueba congeladas del experimento A (data/particion.csv), que son las
etiquetas de las que dependen los resultados: POR_CLASE fotos de cada mueble, más de contenido ajeno y todas las
capturas inválidas, que son pocas. La segunda persona las ve sin clase, con nombres al azar (la carpeta de origen
lleva la clase) y mezcladas. Lo que deje sin clasificar no cuenta.

La auditoría mide la calidad de las etiquetas; no las cambia, porque la partición ya está congelada. Si algún día
se quiere corregir con ella, el autor arbitra los desacuerdos (sort_gui sobre data/auditoria/desacuerdos.csv, con
salida data/auditoria/arbitraje) y `python -m tesis.auditoria aplicar` los lleva a la revisión; después hay que
reconstruir y volver a congelar (tesis.build_dataset, tesis.split --rehacer) y entrenar de nuevo.
"""
import csv
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import CLASSES, SEED
from .metrics import wilson
from .sort_gui import BLIND, copy_name, place

DIR = Path("data/auditoria")
POR_CLASE = {"contenido_ajeno": 150, "captura_invalida": 10**6}  # el resto, 100


def muestra():
    m = pd.read_csv("data/particion.csv", keep_default_na=False)
    m = m[m.A.isin(["val", "test"])]  # solo fotos verificadas, por construcción
    s = pd.concat([g.sample(min(len(g), POR_CLASE.get(c, 100)), random_state=SEED) for c, g in m.groupby("label")])
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    (DIR / "fotos").mkdir(parents=True, exist_ok=True)
    s["ciega"] = [(DIR / "fotos" / f"{i:04d}{Path(p).suffix}").as_posix() for i, p in enumerate(s.path)]
    for src, dst in zip(s.path, s.ciega):
        Path(dst).unlink(missing_ok=True)
        os.link(src, dst)  # enlace duro: no ocupa espacio
    s[["ciega", "path", "label", "store", "source", "A"]].to_csv(DIR / "muestra.csv", index=False)
    pd.DataFrame({"path": s.ciega, "label": BLIND, "model_class": "", "conf": 1.0, "top2": "", "top2_conf": 0.0}
                 ).to_csv(DIR / "cola.csv", index=False)
    print(f"{len(s)} fotos en {DIR / 'cola.csv'} ({-(-len(s) // 24)} páginas):", s.label.value_counts().to_dict())


def decisiones(carpeta):
    r = pd.read_csv(DIR / carpeta / "revision.csv", encoding="utf-8-sig").drop_duplicates("path", keep="last")
    return r.set_index("path").label


def informe():
    s = pd.read_csv(DIR / "muestra.csv")
    s["auditor"] = s.ciega.map(decisiones("segunda"))
    d = s[s.auditor.notna() & (s.auditor != BLIND)]
    print(f"clasificadas por la segunda persona: {len(d)} de {len(s)}")
    rows = []
    for c in CLASSES:
        g = d[d.label == c]
        k = int((g.auditor == c).sum())
        rows.append({"clase": c, "fotos": len(g), "coinciden": k, "concordancia": k / len(g) if len(g) else float("nan"),
                     "IC 95 %": "[{:.3f}, {:.3f}]".format(*wilson(k, len(g)))})
    k = int((d.auditor == d.label).sum())
    rows.append({"clase": "todas", "fotos": len(d), "coinciden": k, "concordancia": k / len(d),
                 "IC 95 %": "[{:.3f}, {:.3f}]".format(*wilson(k, len(d)))})
    out = pd.DataFrame(rows)
    # Lo que va al paquete de evidencias (tesis.evidencia): el resumen y cada foto con las dos clases
    out.to_csv("data/auditoria_concordancia.csv", index=False)
    d.assign(foto=d.path.str.rsplit("/", n=1).str[-1]).rename(
        columns={"A": "conjunto", "label": "etiqueta", "auditor": "segunda_persona"})[
        ["foto", "conjunto", "etiqueta", "segunda_persona"]].to_csv("data/auditoria.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"kappa de Cohen: {cohen_kappa_score(d.label, d.auditor):.3f}")
    print(pd.crosstab(d.label, d.auditor, rownames=["etiqueta"], colnames=["segunda persona"]).to_string())
    dis = d[d.auditor != d.label]
    # Para arbitrar: sugerida la clase de la segunda persona; con "a", la etiqueta que tenía
    pd.DataFrame({"path": dis.ciega, "label": dis.auditor, "model_class": "etiqueta: " + dis.label, "conf": 0.5,
                  "top2": dis.label, "top2_conf": 0.5}).to_csv(DIR / "desacuerdos.csv", index=False)
    print(f"{len(dis)} desacuerdos en {DIR / 'desacuerdos.csv'}")


def aplicar():
    """Lleva cada arbitraje que cambia la etiqueta a la revisión de la que salió la foto, como si se hubiera
    corregido en tesis.sort_gui: mueve la copia a la carpeta de la clase nueva y agrega la fila a revision.csv."""
    s = pd.read_csv(DIR / "muestra.csv")
    s["final"] = s.ciega.map(decisiones("arbitraje"))
    cambios = s[s.final.notna() & (s.final != s.label)]
    origen = {}  # nombre de la copia -> (carpeta de la revisión, ruta original)
    for log in Path("data").glob("clasificadas*/revision.csv"):
        for p in pd.read_csv(log, encoding="utf-8-sig").path.unique():
            origen[copy_name(p)] = (log.parent, p)
    for x in cambios.itertuples():
        out, path = origen[Path(x.path).name]
        place(out, path, x.label, x.final)
        with (out / "revision.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([path, x.final, x.label])
    print(f"{len(cambios)} etiquetas corregidas de {s.final.notna().sum()} arbitradas:",
          cambios.groupby(["label", "final"]).size().to_dict(), "· ahora: python -m tesis.build_dataset")


if __name__ == "__main__":
    {"muestra": muestra, "aplicar": aplicar}.get(sys.argv[1] if len(sys.argv) > 1 else "", informe)()
