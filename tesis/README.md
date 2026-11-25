# `tesis/`: código de la tesis

Paquete de Python. Cada módulo se corre desde la raíz del repositorio con `python -m tesis.<módulo>` y explica en su
docstring cómo se usa y por qué está hecho así. El paquete es plano y sus nombres son fijos porque los cuadernos de
Colab lo importan desde `tesis_colab.zip`.

En el orden en que se usan:

### 1. Etiquetado

| Módulo | Qué hace | Planificación |
|---|---|---|
| [`prelabel.py`](prelabel.py) | Sugiere una clase a cada una de las 64 292 fotos históricas con un ResNet-18 previo | 5.3 |
| [`sort_gui.py`](sort_gui.py) | Revisión manual por páginas: clic o arrastre para seleccionar, tecla o botón para asignar | 5.3 |
| [`odk_fotos.py`](odk_fotos.py) | Descarga, de solo lectura, las fotos de ODK Central (sep–nov 2026) con el mueble declarado | 5.3, 5.9 |
| [`odk_candidatas.py`](odk_candidatas.py) | Ordena las fotos de ODK para revisarlas: SOS y Promociones como contenido ajeno; muebles alquilados, acuerdos comerciales y extravisibilidad, sugeridos como el mueble declarado | 5.3 |
| [`auditoria.py`](auditoria.py) | Auditoría ciega: una muestra por clase que una segunda persona clasifica sin ver la etiqueta, la concordancia con su IC 95 % y el arbitraje de los desacuerdos | 5.3 |

### 2. Conjunto de datos

| Módulo | Qué hace | Planificación |
|---|---|---|
| [`build_dataset.py`](build_dataset.py) | Manifiesto con local, fecha y fuente. Une variantes del nombre de un mismo local y quita fotos sin local, cuasi-duplicados dentro del local y toda foto archivada en dos locales. No mira la partición | 5.1, 5.4 |
| [`degrade.py`](degrade.py) | Capturas inválidas sintéticas (física de sensor y lente, superposición restaurada), filtradas por identificabilidad | 5.3 |
| [`split.py`](split.py) | Partición 70/10/20 agrupada por local sobre el conjunto ya deduplicado, y la lista congelada de fotos de cada partición y experimento (`data/particion.csv`), con la comprobación de que nada se comparte | 5.4 |
| [`convenios.py`](convenios.py) | Convenio de cada local desde las maestras de muebles de Industrias Ales (junio de 2025 y noviembre de 2026), sin usar las etiquetas | 5.6, 5.9 |
| [`export.py`](export.py) | Fotos derechas (EXIF aplicado) a 320 px y el paquete `tesis_colab.zip` para Colab, con la partición congelada y los convenios | 5.5 |

### 3. Entrenamiento y evaluación

Se corren en Colab desde [`notebooks/entrenar_tesis.ipynb`](../notebooks/entrenar_tesis.ipynb).

| Módulo | Qué hace | Planificación |
|---|---|---|
| [`config.py`](config.py) | Clases, máscara de la superposición, semilla y umbrales de la tabla 5.7 | 2.2, 5.4, 5.5, 5.7 |
| [`data.py`](data.py) | Carga de imágenes con `tf.data` y aumentos | 5.5 |
| [`train.py`](train.py) | Aprendizaje por transferencia con protocolo común (MobileNetV3-Large, EfficientNet-B0, ResNet-50), con la máscara dentro del modelo | 5.5 |
| [`quantize.py`](quantize.py) | Conversión a TFLite: float32, int8 calibrado, float16, int8 de rango dinámico e int8 con calibración balanceada | 5.5 |
| [`evaluate.py`](evaluate.py) | Predicciones y métricas de un run sobre su prueba, en Keras o TFLite; con `--val`, las predicciones sobre la validación | 5.6 |
| [`metrics.py`](metrics.py) | F1-macro, recall de cada rechazo con su intervalo de Wilson, decisión del verificador contra el convenio (restringida, sin restringir y con el umbral calibrado en validación), falsas alarmas, bootstrap por local y criterios de la tabla 5.7 | 5.6–5.9 |
| [`evidencia.py`](evidencia.py) | Paquete de evidencias sin fotos (manifiesto, partición, convenios, revisiones, corridas, mediciones), su versión reducida sin modelos, y la tabla de resultados recalculada desde las predicciones y comparada con la reportada | 5.6–5.9 |

### 4. Teléfono

| Módulo | Qué hace | Planificación |
|---|---|---|
| [`device.py`](device.py) | Latencia y memoria en el Honor X6b por adb: el modelo solo (A7) y el prototipo de extremo a extremo (A8). Es un solo archivo con la biblioteca estándar, para correrlo en cualquier computador | 5.6, 5.7 |
| [`fotos_app.py`](fotos_app.py) | Elige las fotos reales de ODK con que se mide el prototipo, compara sus predicciones con las de la evaluación, emula en Python el preprocesamiento del teléfono para ver de dónde sale cada diferencia, cuenta cuántas decisiones de aceptar o rechazar y cuántas alertas cambia en la prueba, y arma el paquete para medir desde otro computador | 5.6 |

## Datos

No se versionan, por volumen y por confidencialidad de Industrias Ales. El conjunto se reconstruye en `data/` en este
orden:

| Paso | Comando |
|---|---|
| 1 | `tesis.prelabel` (con el entorno del clasificador previo, que usa PyTorch) |
| 2 | `tesis.sort_gui data/prelabel.csv data/clasificadas` |
| 3 | `tesis.odk_fotos`, `tesis.odk_candidatas` y `tesis.sort_gui` sobre sus candidatas, para sumar las fotos de ODK. *Fotos mal tomadas* no registra el local: se lee de la superposición |
| 4 | `tesis.build_dataset` (deduplica) |
| 5 | `tesis.degrade 2600`, el filtro de identificabilidad y otra vez `tesis.build_dataset` |
| 6 | `tesis.auditoria` (muestra, segunda persona, arbitraje, aplicar) y otra vez `tesis.build_dataset` |
| 7 | `tesis.split data/manifest.csv data/stores_split.csv`: sortea los locales y congela `data/particion.csv`. Desde aquí la validación y la prueba no cambian: si cambiarían, el comando falla |
| 8 | `tesis.convenios` (con las maestras en `data/convenios/`) |
| 9 | `tesis.export`, y se sube `data/tesis_colab.zip` a `MyDrive/tesis` |

`data/manifest.csv` tiene una fila por imagen: `path, label, store, date, source, verified, declarado`. `source` es
`hist` (fotografía histórica), `odk` (ODK Collect, sep–nov 2026, con el mueble `declarado` en el formulario) o `sint`
(sintética). La validación y la prueba usan solo filas con `verified=1`. `data/particion.csv` agrega a cada foto la
partición de su local (`split`) y su papel en los experimentos A y B; `tesis.train` lee esa lista.
