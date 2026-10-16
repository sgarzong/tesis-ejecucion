# Clasificación de muebles de exhibición en dispositivos de gama baja

Código experimental de la tesis de Maestría en Ciencia de Datos (USFQ), de Sebastián Garzón.
El protocolo completo está en [planificacion_titulacion_reconocimiento_pos_v4.md](planificacion_titulacion_reconocimiento_pos_v4.md).

Modelos: ResNet-50 (línea base), MobileNetV3-Large y EfficientNet-B0, con transferencia desde ImageNet,
cuantización post-entrenamiento a int8 con TFLite y 7 clases (5 muebles y 2 de rechazo).

## Estructura

| Archivo | Qué hace |
|---|---|
| `tesis/config.py` | Clases, umbrales de la tabla 5.7 y valores de calibración (máscara de la superposición, tope por local) |
| `tesis/split.py` | Partición agrupada por local (exp. A), temporal (exp. B), huella perceptual y cuasi-duplicados |
| `tesis/data.py` | Carga con `tf.data` y aumentos de captura en tienda |
| `tesis/train.py` | Entrenamiento con protocolo común; la máscara de la superposición va dentro del modelo |
| `tesis/quantize.py` | Exportación a TFLite en float32 e int8 |
| `tesis/evaluate.py` | Métricas sobre prueba: F1-macro, recall, matriz de confusión, exactitud condicionada y verificación |
| `tesis/metrics.py` | Métricas, IC bootstrap por local y chequeo de viabilidad |

## Datos (no versionados)

`data/manifest.csv`, una fila por imagen:

```
path,label,store,date,source,verified
data/img/0001.jpg,chiclera,TIA LA MARIN,2024-06-12,hist,1
```

- `store`: nombre **normalizado** del local, que es la clave de agrupamiento.
- `label`: uno de `config.CLASSES`, o vacío si la imagen no tiene etiqueta.
- `verified`: 1 si la etiqueta fue revisada a mano. El conjunto de prueba usa solo estas imágenes.

`data/contracts.csv` (opcional, para la exactitud condicionada y la verificación): `store,label`, con una fila por convenio.

## Flujo

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt

python -m tesis.split data/manifest.csv data/stores_split.csv     # asigna locales a train/val/test (idempotente)
python -m tesis.train --model mobilenetv3 --exp A                  # -> runs/A_mobilenetv3/
python -m tesis.train --model mobilenetv3 --exp A --no-mask        # control del atajo de la superposición
python -m tesis.quantize runs/A_mobilenetv3                        # model_fp32.tflite, model_int8.tflite
python -m tesis.evaluate runs/A_mobilenetv3 model_int8.tflite --contracts data/contracts.csv

python -m pytest tests
```

`data/stores_split.csv` es el registro de la partición. Respáldalo: si se borra, la asignación cambia y se rompe la consistencia entre corpus.

TensorFlow no tiene soporte de GPU en Windows nativo, así que el entrenamiento se corre en WSL2 o en la nube.
