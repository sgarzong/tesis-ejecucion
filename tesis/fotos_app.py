"""Fotos para medir el prototipo Android de extremo a extremo (A8) y comprobar que clasifica igual que la evaluación.

    python -m tesis.fotos_app                                 # elige las fotos: data/fotos_app/ y data/fotos_camara/
    python -m tesis.fotos_app runs/final/A_mobilenetv3/model_fp16.tflite runs/final/fotos_app_por_foto.csv   # paridad
    python -m tesis.fotos_app --paquete [destino.zip]     # para medir desde otro computador: APK, device.py, fotos
    python -m tesis.fotos_app --diferencias runs/final/A_mobilenetv3/model_fp16.tflite   # de dónde sale cada diferencia
    python -m tesis.fotos_app --emular runs/final/A_mobilenetv3    # la prueba con el preprocesamiento del teléfono

Las fotos son 120 de ODK de locales de prueba del experimento A, tal como las guarda ODK (≈1024 px, casi todas con
la etiqueta EXIF de rotación): las capturas inválidas reales y el resto repartido entre las demás clases. Van al
teléfono con python -m tesis.device runs/final --app <apk> --fotos data/fotos_app, que escribe fotos_app_por_foto.csv
con las probabilidades que el teléfono da a cada una.

Las capturas inválidas que salieron directo de la cámara (las que tomó el autor, a resolución completa) van aparte, a
data/fotos_camara: son otra medición, la de la foto que la app recibe cuando se abre desde ODK, y se corren igual
con --fotos data/fotos_camara.

Paridad: el teléfono parte de la foto original (EXIF, 320 px y 224 × 224 en Java); la evaluación, de la copia de
tesis.export (PIL) con tf.image.resize. Se comparan la clase y las probabilidades del mismo .tflite en los dos.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import CLASSES, IMG_SIZE, REJECT

EXPORT = 320  # tesis.export.SIDE
OUT = Path("data/fotos_app")
CAMARA = Path("data/fotos_camara")
LADO_ODK = 1100  # px del lado mayor: hasta aquí la foto es como la guarda ODK; más grande, salió directo de la cámara
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
     si no está, baja la carpeta A_mobilenetv3 de MyDrive/tesis/runs_final y descomprímela dentro de modelos/.
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

3. Sube los seis .csv a MyDrive/tesis/runs_final.

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
    t = m[(m.source == "odk") & (m.verified == 1) & m.store.isin(test)
          & m.path.str.lower().str.endswith((".jpg", ".jpeg"))]  # tesis.device solo lleva JPEG al teléfono
    inv, resto = t[t.label == "captura_invalida"], t[t.label != "captura_invalida"]
    grande = np.array([max(Image.open(p).size) > LADO_ODK for p in inv.path])
    camara, inv = inv[grande], inv[~grande]
    por_clase = -(-(N - len(inv)) // resto.label.nunique())
    sel = pd.concat([inv, resto.groupby("label").sample(por_clase, random_state=0)]).head(N)
    sel = pd.concat([sel.assign(carpeta=OUT.name), camara.assign(carpeta=CAMARA.name)])
    sel["archivo"] = [Path(p).name for p in sel.path]
    assert sel.archivo.is_unique
    for d in (OUT, CAMARA):
        assert not any(d.glob("*.jpg")), f"{d} tiene las fotos de otra selección: muévela o bórrala"
        d.mkdir(parents=True, exist_ok=True)
    for p, a, c in zip(sel.path, sel.archivo, sel.carpeta):
        shutil.copy2(p, OUT.parent / c / a)
    sel.to_csv(OUT / "fotos.csv", index=False)  # las dos carpetas: la paridad busca cada foto aquí
    print(sel.groupby("carpeta").label.value_counts().unstack(fill_value=0).to_string())


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


def _bilineal(a, alto, ancho):
    import tensorflow as tf
    # Sin antialias y a 8 bits, como Bitmap.createScaledBitmap con filtro; a la mitad exacta es el promedio de 2 × 2
    return np.clip(np.round(tf.image.resize(a, (alto, ancho), "bilinear").numpy()), 0, 255)


# ponytail: solo los giros (EXIF 3, 6 y 8); las orientaciones en espejo no aparecen en estas fotos
GIRO = {3: lambda a: np.rot90(a, 2), 6: lambda a: np.rot90(a, -1), 8: np.rot90}


def como_telefono(path, girar_antes=False, lanczos=False):
    """La entrada de la red como la arma Clasificador.java (reducir y clasificar), paso a paso en Python, para ver de
    dónde salen sus diferencias con la evaluación. Las dos opciones cambian un paso por el de la evaluación:
    `girar_antes` endereza la foto antes de reducirla y `lanczos` la reduce a 320 px con el filtro de tesis.export."""
    im = Image.open(path)
    girar = GIRO.get(im.getexif().get(274), lambda a: a)
    e = min(im.size) // (2 * EXPORT)
    s = 8 if e >= 8 else 4 if e >= 4 else 2 if e >= 2 else 1
    if s > 1:
        im.draft("RGB", (-(-im.width // s), -(-im.height // s)))  # la escala de decodificación JPEG de inSampleSize
    a = np.asarray(im.convert("RGB"), np.float32)
    if girar_antes:
        a = girar(a)
    f = EXPORT / max(a.shape[:2])
    h, w = int(a.shape[0] * f + .5), int(a.shape[1] * f + .5)
    if lanczos:
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS, reducing_gap=2.0),
                       np.float32)
    else:
        while a.shape[1] >= 2 * w and a.shape[0] >= 2 * h:
            a = _bilineal(a, a.shape[0] // 2, a.shape[1] // 2)
        a = _bilineal(a, h, w)
    return _bilineal(a if girar_antes else girar(a), IMG_SIZE, IMG_SIZE)


def sin_recomprimir(path):
    """La entrada de la red como en la evaluación (tesis.export.upright y tesis.data), pero sin guardar la copia de
    320 px como JPEG: aísla el efecto de esa recompresión."""
    im = Image.open(path)
    im.draft("RGB", (EXPORT * 2, EXPORT * 2))
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((EXPORT, EXPORT), Image.Resampling.LANCZOS)
    return _bilineal(np.asarray(im, np.float32), IMG_SIZE, IMG_SIZE)


def red(modelo):
    """Función entrada (224, 224, 3) -> probabilidades, con el .tflite."""
    import tensorflow as tf
    it = tf.lite.Interpreter(str(modelo))
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def f(x):
        it.set_tensor(i, x[None].astype(np.float32))
        it.invoke()
        return it.get_tensor(o)[0].copy()
    return f


def diferencias(modelo, runs="runs/final"):
    """De dónde salen las diferencias entre el teléfono y la evaluación. Compara las probabilidades que dio el
    teléfono (<runs>/fotos_*_por_foto.csv) con las de la evaluación y con las de cuatro variantes del
    preprocesamiento, cada una con un paso cambiado. La variante que más se acerca al teléfono dice qué paso
    explica la diferencia. Escribe <runs>/fotos_diferencias.csv, una fila por foto."""
    from .evaluate import predict

    f = red(modelo)
    variantes = {"evaluación sin recomprimir": sin_recomprimir, "teléfono emulado": como_telefono,
                 "emulado, girando antes": lambda p: como_telefono(p, girar_antes=True),
                 "emulado, con Lanczos": lambda p: como_telefono(p, lanczos=True)}
    rechazo, partes = np.isin(CLASSES, list(REJECT)), []
    for carpeta, g in pd.read_csv(OUT / "fotos.csv").groupby("carpeta"):
        tel = pd.read_csv(Path(runs) / f"{carpeta}_por_foto.csv").drop_duplicates("archivo").set_index("archivo")
        t = tel.loc[g.archivo, [f"p_{c}" for c in CLASSES]].to_numpy()
        d = pd.DataFrame({"carpeta": carpeta, "archivo": g.archivo.values, "etiqueta": g.label.values,
                          "lado_px": [max(Image.open(p).size) for p in g.path],
                          "exif": [Image.open(p).getexif().get(274, 1) for p in g.path],
                          "teléfono": np.array(CLASSES)[t.argmax(1)], "p_max_teléfono": t.max(1).round(3)})
        ps = {"evaluación": predict(Path(modelo), g.assign(path=g.path320))}
        ps |= {n: np.array([f(v(p)) for p in g.path]) for n, v in variantes.items()}
        for n, p in ps.items():
            d[n] = np.array(CLASSES)[p.argmax(1)]
            d[f"{n}: dif"] = np.abs(p - t).max(1).round(4)
            d[f"{n}: misma decisión"] = rechazo[p.argmax(1)] == rechazo[t.argmax(1)]  # rechazar o aceptar la foto
        partes.append(d)
        print(f"\n{carpeta}: {len(d)} fotos, lado mayor de {d.lado_px.min()} a {d.lado_px.max()} px, "
              f"orientación EXIF {d.exif.value_counts().to_dict()}")
        print(pd.DataFrame([{"variante": n, "misma clase que el teléfono": int((d[n] == d["teléfono"]).sum()),
                             "misma decisión": int(d[f"{n}: misma decisión"].sum()),
                             "dif mediana": d[f"{n}: dif"].median(), "dif máxima": d[f"{n}: dif"].max()}
                            for n in ps]).round(3).to_string(index=False))
    d = pd.concat(partes)
    d.to_csv(Path(runs) / "fotos_diferencias.csv", index=False)
    otra = d[d["evaluación"] != d["teléfono"]]
    print(f"\nfotos en que el teléfono y la evaluación dan otra clase: {len(otra)} de {len(d)}; "
          f"cambia la decisión de rechazar o aceptar en {int((~otra['evaluación: misma decisión']).sum())}")
    print(otra[["carpeta", "etiqueta", "lado_px", "exif", "p_max_teléfono", "teléfono", "evaluación", *variantes]]
          .to_string(index=False))
    print(f"probabilidad máxima del teléfono: mediana {d.p_max_teléfono.median():.3f} en todas, "
          f"{otra.p_max_teléfono.max():.3f} como mucho en las que difieren; con p ≥ 0.5 hay "
          f"{int((d.p_max_teléfono >= .5).sum())} fotos y difieren {int((otra.p_max_teléfono >= .5).sum())}")


def emular(run, formato="model_fp16"):
    """Toda la prueba del run con el preprocesamiento del teléfono emulado (como_telefono), para contar en cuántas
    fotos cambiaría la clase o la decisión respecto de la evaluación. No es una medición en el teléfono: es la
    emulación que `diferencias` contrasta con él. Deja <run>/telefono_emulado_<formato>.npy, en el orden de test.csv."""
    run = Path(run)
    test = pd.read_csv(run / "test.csv")
    original = dict(zip(pd.read_csv("data/manifest_320.csv").path, pd.read_csv("data/manifest.csv").path))
    f = red(run / f"{formato}.tflite")
    np.save(run / f"telefono_emulado_{formato[6:]}.npy", np.array([f(como_telefono(original[p])) for p in test.path]))
    cambios(run, formato)


def cambios(run, formato="model_fp16"):
    """Fotos de la prueba en que la emulación del teléfono y la evaluación dan otra clase u otra decisión."""
    run = Path(run)
    test = pd.read_csv(run / "test.csv")
    ev, tel = np.load(run / f"probs_{formato}.npy"), np.load(run / f"telefono_emulado_{formato[6:]}.npy")
    y, rechazo = test.label.map(CLASSES.index).to_numpy(), np.isin(CLASSES, list(REJECT))
    otra = ev.argmax(1) != tel.argmax(1)
    decision = rechazo[ev.argmax(1)] != rechazo[tel.argmax(1)]
    print(f"{run.name}, {formato}: {len(test)} fotos de prueba; otra clase en {otra.sum()} ({otra.mean():.2%}); "
          f"otra decisión de rechazar o aceptar en {decision.sum()} ({decision.mean():.2%})")
    print(f"exactitud: evaluación {(ev.argmax(1) == y).mean():.4f}, teléfono emulado {(tel.argmax(1) == y).mean():.4f}")
    print(f"de las {decision.sum()}: la emulación acepta lo que la evaluación rechaza en "
          f"{(decision & rechazo[ev.argmax(1)]).sum()} y rechaza lo que la evaluación acepta en "
          f"{(decision & ~rechazo[ev.argmax(1)]).sum()}; la etiqueta es de rechazo en {(decision & rechazo[y]).sum()}")
    print("otra decisión por etiqueta:", test.label[decision].value_counts().to_dict())
    for c in sorted(REJECT):
        m = y == CLASSES.index(c)
        print(f"recall de {c}: evaluación {(ev.argmax(1)[m] == y[m]).sum()} de {m.sum()}, "
              f"teléfono emulado {(tel.argmax(1)[m] == y[m]).sum()}")
    print(f"fotos de rechazo rechazadas con una u otra categoría, de {rechazo[y].sum()}: evaluación "
          f"{(rechazo[ev.argmax(1)] & rechazo[y]).sum()}, teléfono emulado {(rechazo[tel.argmax(1)] & rechazo[y]).sum()}")
    print(f"probabilidad máxima de la evaluación: mediana {np.median(ev.max(1)):.3f} en todas y "
          f"{np.median(ev.max(1)[otra]):.3f} en las que cambian de clase; con p ≥ 0.9 hay {(ev.max(1) >= .9).sum()} "
          f"fotos y cambian {(otra & (ev.max(1) >= .9)).sum()}")
    # Aceptar o rechazar no es lo mismo que alertar: la alerta compara la decisión con el mueble declarado, así que
    # también cambia si la foto pasa de un mueble a otro. Se mide sobre las fotos de ODK que declaran un mueble, con
    # la decisión con umbral, como odk_umbral_* de tesis.metrics.evaluate_run.
    from .evidencia import umbral
    from .metrics import IDX, MUEBLES, contract_keys, contracts, decide
    conv = pd.read_csv("data/convenios.csv")
    tau, con, keys = umbral(run, conv), contracts(conv), contract_keys(test, conv)
    m = ((test.source == "odk") & test.declarado.isin(MUEBLES)).to_numpy()
    dec = test.declarado[m].map(IDX).to_numpy()
    a_ev, a_tel = (decide(p[m], keys[m], con, tau) != dec for p in (ev, tel))
    ok = y[m] == dec
    print(f"alertas en las {m.sum()} fotos de ODK con un mueble declarado (τ = {tau:.2f}): cambian en "
          f"{(a_ev != a_tel).sum()}; de ellas {(a_ev != a_tel)[ok].sum()} muestran lo declarado y "
          f"{(a_ev != a_tel)[~ok].sum()} son errores de campo; alertas: evaluación {a_ev.sum()}, emulación {a_tel.sum()}")
    for i in np.flatnonzero(a_ev != a_tel):
        j = np.flatnonzero(m)[i]
        print(f"   {test.label.iat[j]}, declarada {test.declarado.iat[j]}: evaluación {CLASSES[ev[j].argmax()]} "
              f"({ev[j].max():.2f}, {'alerta' if a_ev[i] else 'sin alerta'}), emulación {CLASSES[tel[j].argmax()]} "
              f"({tel[j].max():.2f}, {'alerta' if a_tel[i] else 'sin alerta'})")


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
    elif sys.argv[1] in ("--diferencias", "--emular", "--cambios"):
        {"--diferencias": diferencias, "--emular": emular, "--cambios": cambios}[sys.argv[1]](*sys.argv[2:])
    else:
        paridad(*sys.argv[1:3])
