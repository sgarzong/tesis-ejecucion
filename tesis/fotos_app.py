"""Fotos para medir el prototipo Android de extremo a extremo (A8) y comprobar que clasifica igual que la evaluación.

    python -m tesis.fotos_app                                           # elige las fotos: data/fotos_app/
    python -m tesis.fotos_app runs/odk/A_mobilenetv3/model_fp16.tflite runs/odk/app_fotos.csv     # paridad

Las fotos son 120 de ODK de locales de prueba del experimento A, tal como las guarda ODK (≈1024 px, casi todas con
la etiqueta EXIF de rotación): las capturas inválidas reales y el resto repartido entre las demás clases. Van al
teléfono con python -m tesis.device runs/odk --app <apk> --fotos data/fotos_app, que escribe app_fotos.csv con las
probabilidades que el teléfono da a cada una.

Paridad: el teléfono parte de la foto original (EXIF, 320 px y 224 × 224 en Java); la evaluación, de la copia de
tesis.export (PIL) con tf.image.resize. Se comparan la clase y las probabilidades del mismo .tflite en los dos.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES

OUT = Path("data/fotos_app")
N = 120


def elegir():
    m = pd.read_csv("data/manifest.csv")
    m["path320"] = pd.read_csv("data/manifest_320.csv").path  # tesis.export conserva el orden de las filas
    test = set(pd.read_csv("data/stores_split.csv").query("split == 'test'").store)
    t = m[(m.source == "odk") & (m.verified == 1) & m.store.isin(test)]
    inv, resto = t[t.label == "captura_invalida"], t[t.label != "captura_invalida"]
    por_clase = -(-(N - len(inv)) // resto.label.nunique())
    sel = pd.concat([inv, resto.groupby("label").sample(por_clase, random_state=0)]).head(N)
    sel["archivo"] = [Path(p).name for p in sel.path]
    assert sel.archivo.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    for p, a in zip(sel.path, sel.archivo):
        shutil.copy2(p, OUT / a)
    sel.to_csv(OUT / "fotos.csv", index=False)
    print(f"{len(sel)} fotos -> {OUT}: {sel.label.value_counts().to_dict()}")


def paridad(modelo, app_csv):
    from .evaluate import predict

    d = pd.read_csv(OUT / "fotos.csv").merge(pd.read_csv(app_csv).drop_duplicates("archivo"), on="archivo")
    pc = predict(Path(modelo), d.assign(path=d.path320))
    tel = d[[f"p_{c}" for c in CLASSES]].to_numpy()
    y = d.label.map(CLASSES.index).to_numpy()
    dif = np.abs(pc - tel).max(1)
    print(f"{len(d)} fotos: misma clase en el teléfono y en la evaluación en {(pc.argmax(1) == tel.argmax(1)).mean():.1%}; "
          f"mayor diferencia de probabilidad por foto: mediana {np.median(dif):.3f}, máxima {dif.max():.3f}")
    print(f"exactitud sobre estas fotos: teléfono {(tel.argmax(1) == y).mean():.1%}, evaluación {(pc.argmax(1) == y).mean():.1%}")
    otra = pc.argmax(1) != tel.argmax(1)
    print(pd.DataFrame({"archivo": d.archivo, "foto": d.label, "teléfono": np.array(CLASSES)[tel.argmax(1)],
                        "evaluación": np.array(CLASSES)[pc.argmax(1)], "dif": dif.round(3)})[otra].to_string(index=False))


if __name__ == "__main__":
    elegir() if len(sys.argv) == 1 else paridad(*sys.argv[1:3])
