# Clasificación de muebles de exhibición en dispositivos de gama baja

Proyecto de Machine Learning y tesis de la Maestría en Ciencia de Datos (USFQ) de Sebastián Garzón.

Los mercaderistas de Industrias Ales fotografían cada mueble de exhibición contratado en las cadenas de autoservicio
(Tía, Favorita, El Rosado, Santa María, Coral) como evidencia del convenio. El modelo clasifica cada foto en **7
clases**, para alertar en el mismo teléfono cuando la evidencia no sirve:

| Clase | Qué muestra |
|---|---|
| `chiclera`, `lateral`, `check_out`, `cabecera`, `exhibidor_piso` | Un mueble contratado |
| `captura_invalida` | Foto inservible: oscura, negra, desenfocada, movida, al piso o al techo |
| `contenido_ajeno` | Otra cosa: la percha regular sin mueble, producto suelto, personas, pantallas |

## Dónde empezar

**[`notebooks/proyecto_ml.ipynb`](notebooks/proyecto_ml.ipynb)** es el entregable del proyecto de ML: introducción,
EDA, modelo por transfer learning (MobileNetV3-Large y EfficientNet-B0 congeladas + regresión logística en un pipeline
de scikit-learn), optimización de λ con Optuna y k-fold repetido **agrupado por local** (curva de sesgo y varianza),
tamaño de entrenamiento m\* según la curva de aprendizaje, comparación estadística 10×10 (Wilcoxon y t-test corregido de Nadeau-Bengio), t-SNE,
evaluación en prueba, conclusiones y referencias IEEE.

Para correrlo en Google Colab:

1. Sube `tesis_colab.zip` (lo genera `python -m tesis.export`) a `MyDrive/tesis` en Google Drive.
2. Abre el cuaderno en Colab, *Entorno de ejecución → GPU*, y *Ejecutar todas*. Monta Drive, descomprime los datos,
   calcula y guarda los rasgos, y corre todo (≈ 1.5 h en una A100).

[`notebooks/entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) hace el ajuste fino de punta a punta del protocolo
de la tesis (3 arquitecturas, experimentos A y B, control sin máscara), la cuantización a TFLite int8 y la evaluación,
el experimento de cuantización para el teléfono (float16, int8 de rango dinámico, int8 con calibración balanceada) y
el análisis según la planificación: tabla 5.7 con las falsas alarmas, verificación contra convenio (pares por
intercambio y sumisiones reales de ODK), exactitud condicionada al local, hipótesis 5.8 y errores por cadena.

La latencia y la memoria en el Honor X6b se miden con `python -m tesis.device runs/odk` (teléfono por USB; los pasos
están en su docstring). [`tesis/device.py`](tesis/device.py) es un solo archivo con la biblioteca estándar: en otro
computador basta Python y `python device.py <carpeta>`, y baja adb si falta. Escribe `dispositivo.csv`, que se sube a
`runs_odk` en Drive para que `entrenar_tesis` complete la tabla 5.7 y dibuje el frente de Pareto.

## Prototipo Android (A8)

[`android/`](android/) es la verificación en el teléfono, sin conexión (objetivo 5): toma la foto, la clasifica con el
`.tflite`, la contrasta con el mueble declarado y advierte si es una captura inválida o contenido ajeno. Java sin más
dependencia que el runtime de TFLite (LiteRT); el preprocesamiento reproduce el del entrenamiento (EXIF, 320 px,
224 × 224) y la decisión, la de `tesis.metrics.conditional_pred`. Se abre sola o desde una pregunta de foto de ODK
Collect, a la que devuelve la foto:

```
type: image   appearance: ex:ec.usfq.tesis.VERIFICAR(mueble=${mueble_nombre}, convenio=${muebles_del_local})
```

`mueble` va en la nomenclatura del formulario (`CHECK OUT | LATERAL SIMPLE | …`); `convenio`, opcional y separado por
comas, restringe la decisión a los muebles del local; `camara`, opcional, fija la app de cámara
(`'com.jeyluta.timestampcamerafree'`).

```powershell
$env:JAVA_HOME = "$env:LOCALAPPDATA\Android\jdk17"; $env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
android\gradlew.bat -p android assembleRelease        # android\app\build\outputs\apk\release\app-release.apk
```

Necesita JDK 17 y el SDK de Android (plataforma 35, build-tools 34). El APK no lleva el modelo: `tesis.device --app`
lo instala y le deja el `.tflite` en el teléfono (para llevarlo dentro del paquete, `app/src/main/assets/modelo.tflite`).
La latencia de extremo a extremo se mide con el mismo protocolo de ráfagas, sobre 120 fotos reales de ODK de locales
de prueba, y se comprueba que el teléfono clasifica igual que la evaluación:

```powershell
python -m tesis.fotos_app                                                          # data/fotos_app/
python -m tesis.device runs/odk --app android\app\build\outputs\apk\release\app-release.apk --fotos data/fotos_app
python -m tesis.fotos_app runs/odk/A_mobilenetv3/model_fp16.tflite runs/odk/fotos_app_por_foto.csv    # paridad
```

En otro computador basta el paquete que arma `python -m tesis.fotos_app --paquete` (el APK, `device.py`, las fotos y
un LEEME), con Python y el teléfono.

## Datos

No se versionan (volumen y confidencialidad de Industrias Ales; ver `.gitignore`). El conjunto se construye en este
orden, y cada paso es reproducible:

| Paso | Comando | Qué hace |
|---|---|---|
| 1 | `tesis.prelabel` | Sugiere una clase a cada una de las 64 292 fotos históricas con un ResNet-18 previo |
| 2 | `tesis.sort_gui` | Revisión manual por páginas (clic o arrastre para seleccionar, tecla o botón para asignar) |
| 3 | `tesis.build_dataset` | Manifiesto con local, fecha y fuente; une variantes del nombre de un mismo local; quita fotos sin local, cuasi-duplicados dentro del local y la misma foto archivada en dos locales (queda la copia que no es de prueba) |
| 4 | `tesis.degrade` | Capturas inválidas sintéticas (física de sensor y lente, superposición restaurada), filtradas por identificabilidad |
| 5 | `tesis.split` | Partición 70/10/20 agrupada por local, estable al llegar datos nuevos (`data/stores_split.csv`, respaldarlo) |
| 6 | `tesis.export` | Fotos derechas (EXIF aplicado) a 320 px y el paquete para Colab |
| — | `tesis.odk_fotos`, `tesis.odk_candidatas` | Fotos de ODK Central (sep–nov 2026) para revisarlas: SOS y Promociones como contenido ajeno; muebles alquilados, acuerdos comerciales y extravisibilidad sugeridas como el mueble declarado; *Fotos mal tomadas*, sin local en el formulario (se lee de la superposición) |

`data/manifest.csv` tiene una fila por imagen: `path, label, store, date, source, verified, declarado`. `source` es
`hist` (fotografía histórica), `odk` (ODK Collect, sep–nov 2026, con el mueble `declarado` en el formulario) o `sint`
(sintética); la prueba usa solo filas con `verified=1`. Las fotos de ODK se unen a los locales del histórico, que
conservan su partición.

## Decisiones de diseño que importan

- **Todo se valida agrupando por local.** El mismo local se fotografía cada mes; separar por foto mezclaría el mismo
  mueble en entrenamiento y prueba.
- **La superposición de Timestamp Camera se enmascara** dentro del modelo (0.70 × 0.33 abajo a la derecha, medido en
  777 fotos): su última línea es el nombre del local, que delata la clase vía el convenio.
- **El 84 % de las fotos viene acostada** con una etiqueta EXIF que TensorFlow ignora; `tesis.export` las endereza.
- **Las capturas inválidas sintéticas heredan el local de su origen** (sin fuga) y nunca entran a la prueba.

## Respecto de la planificación

[`planificacion_titulacion_reconocimiento_pos_v4.md`](planificacion_titulacion_reconocimiento_pos_v4.md) es el
protocolo. Se sigue en la taxonomía de 7 clases, la partición agrupada por local (las fotos de ODK conservan la
asignación de su local), los experimentos A y B, la prueba solo con fotos verificadas, las tres arquitecturas con
protocolo común, el int8 con calibración, la máscara con su control sin máscara, y las métricas y umbrales de la tabla
5.7 (`tesis/config.py`). Desviaciones que la memoria debe declarar:

- **Etiquetado (5.3, objetivo 1).** Revisión manual foto a foto en lugar de etiquetas débiles de la maestra de
  convenios con una estimación de su ruido; sin reconocimiento óptico de la superposición ni reconstrucción de la
  visita. Las 4 198 fotos de SOS y Promociones de ODK entrenan sin revisión (`verified=0`, fuera de la prueba), contra
  la verificación previa que pide 5.1.
- **Taxonomía (2.2).** ARETE → `lateral`; CHECK OUT | LATERAL → `check_out`; MUEBLE GRIS, FOUR WAY y BIOMBO →
  `exhibidor_piso`; OTA y counter de las cadenas modernas → `contenido_ajeno`, en lugar de dejarlos fuera de alcance.
- **Convenios (5.6, 5.9).** Sin la maestra en el proyecto, el convenio de cada local se aproxima con los muebles de sus
  fotos verificadas y los declarados en ODK (`tesis.metrics.contracts`). Es un superconjunto, así que las falsas
  alarmas y la exactitud condicionada son conservadoras.
- **Deduplicación (5.4).** Cuasi-duplicados dentro del local (dhash ≤ 4) en lugar de la visita reconstruida. Entre
  locales, solo la misma foto (hash idéntico): se agregó al encontrar 92 fotos de prueba repetidas en otro local, 88 de
  ellas en desarrollo. Los runs ya entrenados las excluyen al evaluar.
- **Aumentos (5.5).** El cambio de perspectiva se aproxima con zoom y traslación.
- **Máscara (5.5).** Cubre 0.70 × 0.33 ≈ 23 % de la imagen.
- **Tope por local.** `MAX_PER_STORE = 300` no se calibró.
- **Cuantización (5.5).** Además del int8 obligatorio, float16, int8 de rango dinámico e int8 con calibración
  balanceada.
- **Verificación (5.9).** ODK ya acumula errores reales de campo, así que la validación con discrepancias reales, que
  la planificación dejaba como trabajo futuro, se adelantó.

- **Capturas inválidas reales (5.3).** La prueba A tiene 18: dos de la revisión y 16 fotos oscuras de ODK de locales
  de prueba, aprobadas después de entrenar (`entrenar_tesis` las predice con cada run y las suma). Las 14 del
  formulario *Fotos mal tomadas* (7 capturas inválidas y 7 de contenido ajeno según 2.2; dos más, de farmacias, quedan
  fuera de alcance) son de locales de entrenamiento: entrenan en el próximo reentrenamiento y, mientras, sirven de
  comprobación de campo.

- **Latencia (5.6).** `tesis.device` mide con `benchmark_model` de TFLite el modelo solo, con entrada aleatoria: la
  decodificación y el redimensionamiento de la foto, y el aumento del paquete, se miden con el prototipo. Cada ráfaga
  de 30 corre en un proceso nuevo y la herramienta da el p95 por ráfaga: se reporta el mayor de las cuatro, que acota
  por arriba el de las 120 ejecuciones.

Medido en el Honor X6b (A7, `runs_odk/dispositivo.csv`): todas las configuraciones cumplen la latencia y la memoria
(p95 ≤ 246 ms, ≤ 219 MB), así que la viabilidad depende del recall de rechazo, que ninguna alcanza todavía.
Pendiente: correr el prototipo (A8) en el Honor X6b, con la latencia de extremo a extremo, la paridad con la
evaluación y el aumento del paquete, y probarlo desde un formulario de ODK de prueba.

## Estructura

```
notebooks/   proyecto_ml.ipynb (entregable), entrenar_tesis.ipynb (ajuste fino y TFLite)
tesis/       código: datos, partición, entrenamiento, cuantización, métricas, medición en el teléfono, etiquetado
android/     prototipo de verificación en el teléfono (A8)
tests/       python -m pytest tests
planificacion_titulacion_reconocimiento_pos_v4.md   protocolo completo de la tesis
```

## Instalación local

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
python -m pytest tests
```

TensorFlow no usa GPU en Windows nativo: el entrenamiento se corre en Colab (o WSL2).

## Uso de IA

El código se escribió con asistencia de Claude Code (Anthropic). Los prompts que originaron cada parte están como
comentarios `# Prompt (Claude Code): ...` en las celdas del cuaderno; las decisiones de diseño están justificadas en el
cuaderno y en los docstrings de cada módulo.
