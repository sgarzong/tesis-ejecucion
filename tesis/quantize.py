"""Conversión a TFLite post-entrenamiento (sección 5.5, procedimiento estándar de TFLite).

Uso:
    python -m tesis.quantize runs/A_mobilenetv3              # model_fp32 y model_int8
    python -m tesis.quantize runs/A_mobilenetv3 --variantes  # model_fp16, model_dyn y model_int8bal
    python -m tesis.quantize runs/A_mobilenetv3 --diagnostico  # por qué degrada el int8 (ver diagnostico)
Formatos, de menos a más agresivo:
    fp32     sin cuantizar.
    fp16     pesos en float16; el teléfono calcula en float.
    dyn      pesos en int8, activaciones en float (rango dinámico): no necesita calibración.
    int8     pesos y activaciones en int8, calibrado con 300 imágenes de entrenamiento al azar.
    int8bal  igual, calibrado con 100 imágenes por clase: al azar, casi la mitad son check outs.
La calibración usa imágenes de entrenamiento sin aumentos.
"""
import sys
from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from .config import CLASSES
from .data import dataset

N_CALIB = 300
N_CALIB_CLASE = 100


def saved_model(run):
    saved = run / "saved_model"
    keras.models.load_model(run / "model.keras").export(saved)
    return str(saved)


def converter(saved, optimize=True):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved)
    if optimize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv


def int8(saved, calib):
    def representative():
        for x, _ in dataset(calib, batch=1):
            yield [x]

    conv = converter(saved)
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Entrada y salida quedan en float32: la app entrega RGB [0, 255] sin conocer la escala int8.
    return conv.convert()


def fp16(saved):
    conv = converter(saved)
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def convert(run, variantes=False):
    run = Path(run)
    saved = saved_model(run)
    train = pd.read_csv(run / "train.csv")
    if variantes:
        out = {"model_fp16": fp16(saved),
               "model_dyn": converter(saved).convert(),  # sin representative_dataset: rango dinámico
               "model_int8bal": int8(saved, train.sample(frac=1, random_state=0).groupby("label").head(N_CALIB_CLASE))}
    else:
        out = {"model_fp32": converter(saved, optimize=False).convert(),
               "model_int8": int8(saved, train.sample(N_CALIB, random_state=0))}
    for name, model in out.items():
        (run / f"{name}.tflite").write_bytes(model)
        print(f"{name}.tflite", f"{len(model) / 2**20:.1f} MB")


def diagnostico(run, por_clase=40, n_capas=100):
    """Por qué el int8 completo degrada un modelo (sección 5.5). Sobre una muestra de la prueba (`por_clase` fotos de
    cada clase) compara cada variante con el modelo float: acuerdo (misma clase) y exactitud. Tres preguntas:

    - ¿Es la calibración? El int8 del protocolo (300 fotos al azar) frente a 1 000 fotos. El formato int8bal del
      protocolo ya prueba la calibración balanceada por clase.
    - ¿Dónde está el error? tf.lite.experimental.QuantizationDebugger mide, tensor por tensor, la diferencia entre
      el modelo float y el cuantizado en `n_capas` fotos. `error_pasos` es su RMSE en pasos de cuantización: cerca
      de 0.29 (1/√12) es solo redondeo; mucho más, que el rango calibrado no representa bien ese tensor.
    - ¿Qué operaciones lo causan? Se deja en float cada tipo de operación del modelo, uno por uno, y el 10 % de
      tensores con más error: si el acuerdo vuelve, ahí está la causa; si no vuelve con ninguno, el error no está
      en un punto sino que se acumula a lo largo de la red.

    Escribe <run>/int8_capas.csv (un tensor por fila) e int8_diagnostico.csv (una variante por fila). Sin XNNPACK
    el int8 corre cerca de 1 s por foto en un computador de escritorio: con 40 fotos por clase, alrededor de una
    hora por modelo."""
    import numpy as np

    from .evaluate import predict  # aquí y no arriba: evaluate no hace falta para convertir

    run = Path(run)
    saved = saved_model(run)
    train, test = pd.read_csv(run / "train.csv"), pd.read_csv(run / "test.csv")
    muestra = test.groupby("label").sample(frac=1, random_state=0).groupby("label").head(por_clase)
    y = muestra.label.map(CLASSES.index).values
    ref = predict(run / "model.keras", muestra).argmax(1)

    def rep(df):
        return lambda: ([x] for x, _ in dataset(df, batch=1))

    def conv(calib, estricto=False):
        c = converter(saved)
        c.representative_dataset = rep(calib)
        if estricto:  # como el int8 del protocolo; sin esto, lo que no se cuantiza queda en float
            c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return c

    calib = train.sample(N_CALIB, random_state=0)
    debug = rep(muestra.sample(min(n_capas, len(muestra)), random_state=0))
    dbg = tf.lite.experimental.QuantizationDebugger(converter=conv(calib), debug_dataset=debug)
    # ponytail: atributo privado. XNNPACK no prepara el MobileNetV3 int8 (como en evaluate.predict) y el depurador
    # no deja elegir el intérprete; si una versión de TensorFlow lo renombra, esto falla aquí y no en silencio.
    dbg._quant_interpreter = tf.lite.Interpreter(
        model_content=dbg.quant_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    dbg.run()
    with open(run / "int8_capas.csv", "w") as f:
        dbg.layer_statistics_dump(f)
    capas = pd.read_csv(run / "int8_capas.csv")
    capas["error_pasos"] = np.sqrt(capas.mean_squared_error) / capas.scale
    capas["rango"] = 255 * capas.scale
    capas.sort_values("error_pasos", ascending=False).to_csv(run / "int8_capas.csv", index=False)
    ops = capas.groupby("op_name").error_pasos.agg(tensores="count", mediana="median", maximo="max").sort_values(
        "mediana", ascending=False)
    print(ops.round(2).to_string())

    def selectivo(**kw):
        return tf.lite.experimental.QuantizationDebugger(
            converter=conv(calib), debug_dataset=debug,
            debug_options=tf.lite.experimental.QuantizationDebugOptions(**kw)).get_nondebug_quantized_model()

    variantes = {"int8 del protocolo (300 fotos)": lambda: conv(calib, estricto=True).convert(),
                 "int8 calibrado con 1 000 fotos": lambda: conv(train.sample(min(1000, len(train)), random_state=1),
                                                                estricto=True).convert()}
    for op in ops.index:
        variantes[f"int8 con {op} en float ({ops.tensores[op]} tensores)"] = lambda op=op: selectivo(denylisted_ops=[op])
    n = max(1, len(capas) // 10)
    variantes[f"int8 con los {n} tensores de más error en float"] = lambda: selectivo(
        denylisted_nodes=capas.nlargest(n, "error_pasos").tensor_name.tolist())
    filas = [{"variante": "float32 (referencia)", "MB": (run / "model.keras").stat().st_size / 2**20, "acuerdo": 1.0,
              "exactitud": float((ref == y).mean())}]
    tmp = run / "_diagnostico.tflite"
    for nombre, hacer in variantes.items():
        try:  # una variante que no convierte no debe tirar la hora de las demás
            tmp.write_bytes(hacer())
            p = predict(tmp, muestra).argmax(1)
            filas.append({"variante": nombre, "MB": tmp.stat().st_size / 2**20, "acuerdo": float((p == ref).mean()),
                          "exactitud": float((p == y).mean())})
        except Exception as e:  # noqa: BLE001 (queda escrito en la tabla)
            filas.append({"variante": nombre, "error": f"{type(e).__name__}: {e}"[:200]})
        print(filas[-1])
        pd.DataFrame(filas).to_csv(run / "int8_diagnostico.csv", index=False)
    tmp.unlink(missing_ok=True)
    out = pd.DataFrame(filas)
    print(f"muestra: {len(muestra)} fotos de prueba")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    if "--diagnostico" in sys.argv:
        diagnostico(sys.argv[1])
    else:
        convert(sys.argv[1], "--variantes" in sys.argv)
