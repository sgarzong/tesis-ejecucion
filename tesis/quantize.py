"""Conversión a TFLite post-entrenamiento (sección 5.5, procedimiento estándar de TFLite).

Uso:
    python -m tesis.quantize runs/A_mobilenetv3              # model_fp32 y model_int8
    python -m tesis.quantize runs/A_mobilenetv3 --variantes  # model_fp16, model_dyn y model_int8bal
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


if __name__ == "__main__":
    convert(sys.argv[1], "--variantes" in sys.argv)
