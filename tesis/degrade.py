"""Síntesis controlada de capturas inválidas (sección 5.3): oscuras, negras, desenfocadas y movidas, a partir
de fotos válidas revisadas. Encuadre (piso o techo) y contenido ajeno no se sintetizan: solo hay ejemplares reales.

    python -m tesis.degrade 2600
    <venv_clasificador>\\Scripts\\python -m tesis.prelabel data/sinteticas/captura_invalida data/sinteticas/juez.csv
    python -m tesis.degrade --filtrar data/sinteticas/juez.csv   # quita las que aún se reconocen (~25 %)
    python -m tesis.build_dataset                                # las agrega al manifiesto

Salida: data/sinteticas/captura_invalida/*.jpg y data/sinteticas/sinteticas.csv (path, origen, tipo, param).
build_dataset les asigna el local y la fecha de su foto de origen, así la variante degradada cae en la misma
partición que el original (sin fuga), y verified=0, así el conjunto de prueba solo usa capturas reales.

Criterios:
- Física: la luz se degrada en espacio lineal (sRGB^2.2), como en el sensor. La oscuridad es falta de fotones,
  así que trae ruido de disparo y de lectura que la ganancia automática del teléfono amplifica. El desenfoque
  es un disco (apertura del lente), no una gaussiana. El movimiento es una trayectoria temblorosa, no una recta.
- Severidad: muy por fuera de los aumentos de entrenamiento (brillo ±25 %, contraste ±25 %, data.AUGMENT), para
  que ninguna sintética coincida con una foto válida aumentada. El mueble no debe poder identificarse.
- Superposición: Timestamp Camera estampa el logo (arriba a la derecha) y el texto (abajo a la derecha) después
  de la captura, así que en una foto inválida real se ven nítidos. Se restauran encima de la degradación. Si no,
  el modelo aprendería "logo borroso = inválida", un atajo que falla con las capturas inválidas reales.
- Variedad: fuentes repartidas por igual entre los 5 muebles y, dentro de cada uno, entre locales.
- Identificabilidad: el desenfoque y el movimiento se dejan moderados (realistas y cercanos a la frontera), y
  el ResNet-18 de preetiquetado hace de juez: si aún reconoce el mueble de origen con confianza >= JUDGE_CONF,
  la foto no cumple la definición de captura inválida y se descarta.
"""
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage, signal

from .config import MASK_H, MASK_W, REJECT, SEED, TO_TESIS

OUT = Path("data/sinteticas")
WORK = 1024  # lado largo de trabajo: sobra para 224 px y acota el costo de las convoluciones
MIX = {"oscura": .35, "negra": .05, "desenfocada": .25, "movida": .25, "oscura_movida": .10}
JUDGE_CONF = .8


def to_lin(a):
    return (a / 255.0) ** 2.2


def to_srgb(x):
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


def underexpose(x, rng, light):
    """`light`: fracción de luz que llega al sensor respecto a una exposición correcta (ya con la ganancia)."""
    full_well, gain = 4000, rng.uniform(1, 4)  # el teléfono sube la ganancia: más ruido, igual de oscura
    electrons = rng.poisson(x * light / gain * full_well).astype(float) + rng.normal(0, 3, x.shape)
    return electrons / full_well * gain


def disk_psf(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = (x ** 2 + y ** 2 <= r ** 2).astype(float)
    return k / k.sum()


def shake_psf(length, rng):
    """Paseo aleatorio con inercia, densificado y escalado para que su extensión total sea `length` px."""
    v, pts = rng.normal(size=2), [np.zeros(2)]
    for _ in range(48):
        v = .8 * v + rng.normal(scale=.6, size=2)
        pts.append(pts[-1] + v)
    pts = np.array(pts)
    pts = (pts - pts.mean(0)) * length / max(np.ptp(pts, 0).max(), 1e-6)
    t = np.linspace(0, len(pts) - 1, int(length * 4) + 2)
    dense = np.stack([np.interp(t, np.arange(len(pts)), pts[:, i]) for i in (0, 1)], 1)
    half = int(np.ceil(np.abs(dense).max())) + 1
    k = np.zeros((2 * half + 1,) * 2)
    np.add.at(k, (np.round(dense[:, 1]).astype(int) + half, np.round(dense[:, 0]).astype(int) + half), 1)
    return k / k.sum()


def blur(x, psf):
    out = np.stack([signal.fftconvolve(x[..., c], psf, mode="same") for c in range(3)], -1)
    return np.clip(out, 0, None)  # la FFT deja negativos de redondeo; la luz no es negativa


def overlay_alpha(img):
    """Píxeles de la superposición. Logo: el cuadro blanco más grande arriba a la derecha (unido con un cierre,
    porque las franjas lo cortan) y, si es casi cuadrado, todo su rectángulo con franjas y lente. Texto: píxeles
    casi blancos en la zona de la máscara, dilatados para tomar el antialias y la sombra."""
    h, w = img.shape[:2]
    white = img.min(axis=2) >= 240
    alpha = np.zeros_like(white)
    y1, x0 = int(h * .15), int(w * .75)
    lab, n = ndimage.label(ndimage.binary_closing(white[:y1, x0:], iterations=max(2, w // 80)))
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        ys, xs = np.nonzero(np.isin(lab, np.flatnonzero(sizes >= .3 * sizes.max()) + 1))  # las franjas lo parten
        bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1
        if bw > w * .03 and .7 < bh / bw < 1.4:  # sin logo (o solo una luz del techo) no se restaura nada
            alpha[ys.min():ys.max() + 1, x0 + xs.min():x0 + xs.max() + 1] = True
    text = np.zeros_like(white)
    text[int(h * (1 - MASK_H)):, int(w * (1 - MASK_W)):] = white[int(h * (1 - MASK_H)):, int(w * (1 - MASK_W)):]
    return (alpha | ndimage.binary_dilation(text, iterations=2))[..., None]


def degrade(img, kind, rng):
    """img: uint8 HxWx3. Devuelve (uint8, descripción de parámetros)."""
    x, short = to_lin(img.astype(float)), min(img.shape[:2])
    if kind == "negra":  # falla de la cámara: cuadro negro, sin superposición (como el ejemplar real)
        return to_srgb(np.abs(rng.normal(0, .0004, x.shape))), "sin superposicion"
    if kind in ("movida", "oscura_movida"):
        length = rng.uniform(.09, .18) * short
        x, param = blur(x, shake_psf(length, rng)), f"largo={length / short:.3f}"
    elif kind == "desenfocada":
        r = rng.uniform(.05, .09) * short
        x, param = blur(x, disk_psf(int(r))), f"radio={r / short:.3f}"
    else:
        param = ""
    if kind in ("oscura", "oscura_movida"):
        light = np.exp(rng.uniform(np.log(.0015), np.log(.012)))
        x, param = underexpose(x, rng, light), f"{param} luz={light:.4f}".strip()
    else:
        x = x + rng.normal(0, .002, x.shape)  # ruido base del sensor
    out = to_srgb(x)
    a = overlay_alpha(img)
    return np.where(a, img, out), param


def work(job):
    i, src, kind, dst = job
    rng = np.random.default_rng([SEED, i])
    img = Image.open(src)
    img.draft("RGB", (WORK, WORK))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((WORK, WORK))
    out, param = degrade(np.asarray(img), kind, rng)
    Image.fromarray(out).save(dst, "JPEG", quality=int(rng.integers(35, 76)))  # recompresión de teléfono
    return param


def portrait(path):
    im = Image.open(path)
    w, h = im.size
    return (h > w) != (im.getexif().get(0x0112, 1) in (5, 6, 7, 8))  # EXIF de rotación


def sample(manifest, n):
    """n fuentes por igual entre los muebles y, dentro de cada uno, con peso 1/fotos del local. Solo verticales:
    en las apaisadas (7 %) la superposición cambia de lugar y las zonas de overlay_alpha no aplican."""
    m = manifest[(manifest.source == "hist") & ~manifest.label.isin(REJECT)]
    w = 1 / m.groupby("store").path.transform("size")
    per = n // m.label.nunique()
    rng, picks = np.random.default_rng(SEED), []
    for _, g in m.groupby("label"):
        # Permutación ponderada (Efraimidis-Spirakis); 2x candidatos sobra porque el 93 % son verticales.
        cand = g.loc[(rng.exponential(size=len(g)) / w[g.index]).sort_values().index[:per * 2]]
        picks.append(cand[cand.path.map(portrait)].head(per))
    return pd.concat(picks)


def main(n=2000):
    src = sample(pd.read_csv("data/manifest.csv"), n)
    kinds = np.random.default_rng(SEED).choice(list(MIX), len(src), p=list(MIX.values()))
    shutil.rmtree(OUT / "captura_invalida", ignore_errors=True)  # generadas por este script: se rehacen enteras
    (OUT / "captura_invalida").mkdir(parents=True)
    jobs = [(i, p, k, OUT / "captura_invalida" / f"sint_{k}_{Path(p).name}") for i, (p, k) in enumerate(zip(src.path, kinds))]
    with ProcessPoolExecutor(6) as ex:
        params = list(ex.map(work, jobs, chunksize=8))
    with (OUT / "sinteticas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "origen", "tipo", "param"])
        w.writerows((dst.as_posix(), Path(p).name, k, prm) for (_, p, k, dst), prm in zip(jobs, params))
    print(f"{len(jobs)} sintéticas -> {OUT}")
    print(pd.Series(kinds).value_counts().to_string())


def filter_recognized(judge_csv):
    """Borra las sintéticas en que el juez (salida de tesis.prelabel) aún reconoce el mueble de origen."""
    s = pd.read_csv(OUT / "sinteticas.csv")
    j = pd.read_csv(judge_csv)
    j.index = j.path.map(lambda p: Path(p).as_posix())
    m = pd.read_csv("data/manifest.csv").query("source == 'hist'")
    src = m.set_index(m.path.str.rsplit("/", n=1).str[-1]).label
    hit = (j.loc[s.path, "model_class"].map(TO_TESIS).values == src.reindex(s.origen).values) \
        & (j.loc[s.path, "conf"].values >= JUDGE_CONF)
    for p in s.path[hit]:
        Path(p).unlink(missing_ok=True)
    s[~hit].to_csv(OUT / "sinteticas.csv", index=False)
    print(f"reconocibles (borradas): {hit.sum()} · quedan {(~hit).sum()}")
    print(pd.DataFrame({"tipo": s.tipo, "reconocible": hit}).groupby("tipo").reconocible.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    if sys.argv[1:2] == ["--filtrar"]:
        filter_recognized(Path(sys.argv[2]))
    else:
        main(*(int(a) for a in sys.argv[1:]))
