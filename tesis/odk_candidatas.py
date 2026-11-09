"""Prepara las fotos de ODK (tesis.odk_fotos) como contenido ajeno para revisarlas en tesis.sort_gui.

    <venv_clasificador>\\Scripts\\python -m tesis.odk_candidatas
    python -m tesis.sort_gui data/odk/candidatas.csv data/clasificadas_odk

Las fotos de SOS y Promociones muestran la percha regular, no un mueble: por la sección 2.2 son contenido ajeno,
justo el caso de un mercaderista que fotografía la percha en lugar del mueble. Pero algunas sí muestran un mueble
(una cabecera con la promoción, una ruma). Para ordenarlas se entrena una regresión logística sobre los rasgos
de imagen de CLIP (ViT-B/32): mueble = fotos históricas de los 5 muebles; no mueble = contenido ajeno histórico y
las perchas de SOS. Las descripciones de texto de CLIP no sirven aquí (marcaban como mueble 577 de 659 perchas de
SOS). Es independiente de los modelos de la tesis, así no sesga las etiquetas que se revisan. La sugerencia es
contenido ajeno con confianza = 1 - P(mueble): en la GUI las claras van primero y las que parecen mueble al final.
Alcance (sección 2.2): solo cadenas de autoservicio; fuera farmacias, estaciones de servicio y Plan Ninja
(autoservicios independientes). Una foto oscura se sugiere como captura inválida.

Los formularios de muebles (acuerdos comerciales, extravisibilidad y muebles alquilados) traen el mueble declarado
de cada foto: esa es la sugerencia, llevada a la taxonomía con la tabla de la sección 2.2, y se revisa aparte:

    python -m tesis.sort_gui data/odk/muebles.csv data/clasificadas_odk_muebles

Ahí CLIP (6 clases: los 5 muebles y contenido ajeno) solo ordena: dentro de cada clase, las fotos que no se parecen
a lo declarado van al final con su alternativa ("a: ...?"). Son justo los errores de campo que el modelo debe atrapar.
Fuera: fotos de daño (un detalle, no el mueble) y locales que no son de las cadenas del canal moderno.
"""
import csv
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

CLIP_W = Path(r"C:\Users\sebas\Downloads\proyecto-20260627T133347Z-3-001\proyecto"
              r"\clasificacion_ales\retrain\weights\clip\ViT-B-32.pt")
ODK = Path("data/odk")
FORMS = ("promociones_mercaderista", "sos_participacion_percha_moderno")
OUT_OF_SCOPE = ("Fybeca", "Difare", "Medicity", "Atimasa")  # farmacias y estaciones de servicio
MUEBLES = ["chiclera", "lateral", "check_out", "cabecera", "exhibidor_piso"]
PER_CLASS, DARK_MEAN = 400, 40
MUEBLE_FORMS = ("acuerdos_comerciales", "extravisibilidad_ganados_moderno", "seguimiento_muebles_alquilados_uiux_v4")
CADENAS = r"\bt[ií]a\b|favorita|rosado|santa ?mar[ií]a|coral|liris"  # canal moderno (sección 2.2)
# Mueble declarado -> clase. Tabla de la sección 2.2 (RUMA = exhibidor de piso), más las nomenclaturas de acuerdos
# comerciales y extravisibilidad: el arete es un lateral de cabecera; mueble gris, four way y biombo son exhibidores
# de piso. Lo que no está aquí
# (percha, OTA, exhibidor de counter, ganchera, mueble de frutas y verduras) no es de la taxonomía: contenido ajeno,
# como GANCHERA en config.TO_TESIS, y el revisor decide.
CLASE_DECLARADA = {
    **dict.fromkeys(["CHICLERA", "CHICLERAS"], "chiclera"),
    **dict.fromkeys(["LATERAL SIMPLE", "LATEAL SIMPLE", "LATERAL DOBLE", "LATERAL DE CABECERA",
                     "CHECK OUT LATERAL DOBLE", "ARETE"], "lateral"),
    **dict.fromkeys(["CHECK OUT", "CHECK OUT SIMPLE", "CAJA"], "check_out"),
    **dict.fromkeys(["CABECERA", "CABECERA FRONTAL", "CABECERA SECA", "CABECERA FASTLINE", "MUEBLE GRIS/CABECERA",
                     "BANDEJA DE CABECERA"], "cabecera"),
    **dict.fromkeys(["EXHIBIDOR DE PISO", "RUMA", "RUMAS", "RUMA COMPARTIDA", "MEDIA RUMA", "FOUR WAY", "FOUR WAYS",
                     "MUEBLE GRIS", "MUBLE GRIS", "MUEBLE GRIS COMPARTIDO", "BIOMBO", "BANDEJA DE BIOMBO"],
                    "exhibidor_piso"),
}


def embed(model, pre, paths, cache):
    """Rasgos de CLIP normalizados, con caché en disco (calcularlos es lo lento)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True).item()
        if z["paths"] == list(paths):
            return z["x"], z["brillo"]
    xs, bright = [], []
    for i in range(0, len(paths), 32):
        imgs = []
        for p in paths[i:i + 32]:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((448, 448))
            bright.append(float(np.asarray(im.convert("L"), dtype=np.float32).mean()))
            imgs.append(pre(im))
        with torch.no_grad():
            f = model.encode_image(torch.stack(imgs)).float()
        xs.append((f / f.norm(dim=-1, keepdim=True)).numpy())
        if i % 1280 == 0:
            print(f"  {cache.stem}: {i + len(imgs)}/{len(paths)}", flush=True)
    x = np.concatenate(xs)
    np.save(cache, {"paths": list(paths), "x": x, "brillo": bright})
    return x, bright


def main():
    d = pd.concat([pd.read_csv(ODK / f / "fotos.csv") for f in FORMS], ignore_index=True)
    d = d[~d.cadena.fillna("").str.contains("|".join(OUT_OF_SCOPE))].reset_index(drop=True)
    m = pd.read_csv("data/manifest_320.csv").query("source == 'hist'")
    ref = pd.concat([m[m.label == c].sample(min(PER_CLASS, (m.label == c).sum()), random_state=0) for c in MUEBLES]
                    + [m[m.label == "contenido_ajeno"]], ignore_index=True)

    model, pre = clip.load(str(CLIP_W), device="cpu")
    x_ref, _ = embed(model, pre, ref.path.tolist(), ODK / "clip_ref.npy")
    x_odk, bright = embed(model, pre, d.path.tolist(), ODK / "clip_odk.npy")

    # Entrenamiento: mueble (1) vs no mueble (0). Las perchas de SOS entran como no mueble y se puntúan con
    # validación cruzada agrupada por local, para no calificar una foto con un modelo que ya la vio.
    sos = (d.form == "sos_participacion_percha_moderno").values
    x = np.concatenate([x_ref, x_odk[sos]])
    y = np.concatenate([ref.label.isin(MUEBLES).astype(int), np.zeros(sos.sum(), int)])
    groups = np.concatenate([ref.store.values, ("ODK " + d.local[sos].fillna("")).values])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    cv = cross_val_predict(clf, x, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    acc = ((cv > .5) == y).mean()
    print(f"sondeo lineal mueble/no mueble, validación cruzada por local: exactitud {acc:.3f}"
          f" · perchas SOS marcadas como mueble: {(cv[len(ref):] > .5).mean():.1%}")
    p_mueble = clf.fit(x, y).predict_proba(x_odk)[:, 1]
    p_mueble[sos] = cv[len(ref):]

    d["p_mueble"], d["brillo"] = np.round(p_mueble, 3), np.round(bright, 1)
    d["label"] = np.where(d.brillo < DARK_MEAN, "captura_invalida", "contenido_ajeno")
    d.to_csv(ODK / "candidatas_detalle.csv", index=False)
    with (ODK / "candidatas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "model_class", "conf", "top2", "top2_conf"])
        w.writerows([p, lab, "CLIP", round(1 - pm, 3), "", 0] for p, lab, pm in zip(d.path, d.label, d.p_mueble))
    print(f"{len(d)} fotos · parecen mueble (P >= 0.5): {(d.p_mueble >= .5).sum()} · oscuras: "
          f"{(d.label == 'captura_invalida').sum()}")
    print(d.groupby("form").agg(fotos=("path", "size"), parecen_mueble=("p_mueble", lambda s: (s >= .5).sum())).to_string())
    muebles(model, pre, np.concatenate([x_ref, x_odk[sos]]),
            np.concatenate([ref.label, np.full(sos.sum(), "contenido_ajeno")]))


def muebles(model, pre, x, y):
    """data/odk/muebles.csv: fotos de los formularios de muebles, sugeridas como el mueble declarado."""
    f = pd.concat([pd.read_csv(ODK / form / "fotos.csv") for form in MUEBLE_FORMS], ignore_index=True)
    f = f[f.cadena.fillna("").str.contains(CADENAS, case=False) & ~f.campo.str.endswith("danio_foto")]
    f = f.reset_index(drop=True)
    # Muebles alquilados: "TIPO | VARIANTE | NEGOCIO | ...". El tipo es lo que muestra la foto: "CHECK OUT | LATERAL
    # SIMPLE" es el exhibidor de la caja (así lo etiquetaron las fotos históricas), no el lateral de cabecera.
    f["nombre"] = f.declarado.fillna("").str.split("|").str[0].str.strip().str.upper()

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    x_f, bright = embed(model, pre, f.path.tolist(), ODK / "clip_muebles.npy")
    p = pd.DataFrame(clf.predict_proba(x_f), columns=clf.classes_)
    top = p.columns.to_numpy()[np.argsort(-p.to_numpy(), axis=1)]
    f["brillo"] = np.round(bright, 1)
    f["label"] = np.where(f.brillo < DARK_MEAN, "captura_invalida", f.nombre.map(CLASE_DECLARADA).fillna("contenido_ajeno"))
    f["clip_pred"] = top[:, 0]
    f["conf"] = [round(p.at[i, c], 3) if c in p else .5 for i, c in enumerate(f.label)]
    f["top2"] = np.where(top[:, 0] != f.label, top[:, 0], top[:, 1])  # la alternativa que muestra la GUI
    f["top2_conf"] = [round(p.at[i, c], 3) for i, c in enumerate(f.top2)]
    f.to_csv(ODK / "muebles_detalle.csv", index=False)
    f.assign(model_class="ODK: " + f.nombre)[["path", "label", "model_class", "conf", "top2", "top2_conf"]].to_csv(
        ODK / "muebles.csv", index=False)
    print(f"\nformularios de muebles: {len(f)} fotos de {f.local.nunique()} locales · CLIP coincide con lo declarado en "
          f"{(f.clip_pred == f.label).mean():.0%}")
    print(pd.crosstab(f.label, f.form, margins=True).to_string())


if __name__ == "__main__":
    main()
