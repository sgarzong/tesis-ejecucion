"""Fotos para medir el prototipo Android de extremo a extremo (A8) y comprobar que clasifica igual que la evaluación.

    python -m tesis.fotos_app                                           # elige las fotos: data/fotos_app/
    python -m tesis.fotos_app runs/odk/A_mobilenetv3/model_fp16.tflite runs/odk/fotos_app_por_foto.csv   # paridad
    python -m tesis.fotos_app --paquete [destino.zip]     # para medir desde otro computador: APK, device.py, fotos

Las fotos son 120 de ODK de locales de prueba del experimento A, tal como las guarda ODK (≈1024 px, casi todas con
la etiqueta EXIF de rotación): las capturas inválidas reales y el resto repartido entre las demás clases. Van al
teléfono con python -m tesis.device runs/odk --app <apk> --fotos data/fotos_app, que escribe fotos_app_por_foto.csv
con las probabilidades que el teléfono da a cada una.

Paridad: el teléfono parte de la foto original (EXIF, 320 px y 224 × 224 en Java); la evaluación, de la copia de
tesis.export (PIL) con tf.image.resize. Se comparan la clase y las probabilidades del mismo .tflite en los dos.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES

OUT = Path("data/fotos_app")
N = 120
APK = Path("android/app/build/outputs/apk/release/app-release.apk")
LEEME = """VERIFICADOR DE MUEBLES: MEDICIÓN EN EL TELÉFONO (A8)

Qué hay aquí
  verificador.apk   el prototipo; device.py le instala el modelo
  device.py         instala y mide (Python 3.8 o más reciente, nada más; si falta adb, lo baja de Google)
  fotos_app/        120 fotos reales de ODK de locales de prueba. Son de Industrias Ales: no las compartas.

Antes de empezar
  1. Python 3.8 o más reciente. En Windows: winget install Python.Python.3.12
  2. Una carpeta modelos/ junto a este archivo con A_mobilenetv3/model_fp16.tflite: la de la medición anterior o,
     si no está, baja la carpeta A_mobilenetv3 de MyDrive/tesis/runs_odk y descomprímela dentro de modelos/.
  3. El teléfono: Opciones de desarrollador → Depuración USB y Pantalla activa. Conéctalo por USB, acepta este
     computador en el teléfono y deja ODK Collect abierto. (En Windows, si no aparece, instala HiSuite de Honor.)
  Abre una terminal en esta carpeta (en Windows: clic derecho en la carpeta → Abrir en Terminal).

1. De extremo a extremo con las fotos de ODK (unos 3 minutos)
     python device.py modelos --app verificador.apk --fotos fotos_app
   Deja en modelos/: fotos_app_dispositivo.csv, fotos_app_rafagas.csv y fotos_app_por_foto.csv.

2. Lo mismo con fotos de la cámara a resolución completa (las que recibe la app cuando se abre desde ODK)
   Toma unas 30 fotos de muebles con Timestamp Camera. Pásalas a una carpeta fotos_camara/ junto a este archivo
   (con el teléfono en modo USB «Transferencia de archivos»; están en DCIM) y corre:
     python device.py modelos --app verificador.apk --fotos fotos_camara
   Deja fotos_camara_dispositivo.csv, fotos_camara_rafagas.csv y fotos_camara_por_foto.csv.

3. Sube los seis .csv a MyDrive/tesis/runs_odk.

Probarla a mano
  Después del paso 1 queda instalada con el modelo: «Verificador de muebles». Elige el mueble declarado y toma una
  foto o elige una de la galería. Para solo instalarla: python device.py modelos --app verificador.apk

Desde un formulario de prueba de ODK (no en uno que esté en uso)
  En la pregunta de foto del mueble:
    type: image    appearance: ex:ec.usfq.tesis.VERIFICAR(mueble=${campo_del_mueble})
  Opcionales: convenio=${muebles_del_local}, separados por comas, y camara='com.jeyluta.timestampcamerafree'.
  La app abre la cámara, verifica la foto y la devuelve al formulario con «Usar esta foto».

Si algo falla, copia lo que dice la terminal.
"""


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


def paquete(destino):
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(APK, "verificador/verificador.apk")
        z.write("tesis/device.py", "verificador/device.py")
        for f in sorted(OUT.glob("*.jpg")):
            z.write(f, f"verificador/fotos_app/{f.name}")
        z.writestr("verificador/LEEME.txt", LEEME)
    print(f"{destino}: {destino.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        elegir()
    elif sys.argv[1] == "--paquete":
        paquete(Path(sys.argv[2] if len(sys.argv) > 2 else "data/verificador_telefono.zip"))
    else:
        paridad(*sys.argv[1:3])
