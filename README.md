# Clasificación de muebles de exhibición en dispositivos de gama baja

Tesis de la Maestría en Ciencia de Datos (USFQ) de Sebastián Garzón, y el proyecto de Machine Learning del curso que
se desprende de ella. Tutor: Ricardo Flores Moyano.

Los mercaderistas de Industrias Ales fotografían cada mueble de exhibición contratado en las cadenas de autoservicio
(Tía, Favorita, El Rosado, Santa María, Coral) como evidencia del convenio. El modelo clasifica cada foto en **7
clases** para alertar, en el mismo teléfono, cuando la evidencia no sirve:

| Clase | Qué muestra |
|---|---|
| `chiclera`, `lateral`, `check_out`, `cabecera`, `exhibidor_piso` | Un mueble contratado |
| `captura_invalida` | Una foto inservible: oscura, negra, desenfocada, movida, al piso o al techo |
| `contenido_ajeno` | Otra cosa: la percha regular sin mueble, producto suelto, personas, pantallas |

## Dónde está cada cosa

```
docs/planificacion_v4.md      la planificación aprobada: objetivos, métricas, umbrales (tabla 5.7) y cronograma
notebooks/
  entrenar_tesis.ipynb        la tesis: entrenamiento, cuantización y análisis, con los resultados
  proyecto_ml.ipynb           el proyecto del curso: EDA, transfer learning con regresión logística, validación agrupada
  ajuste_fino.csv             tabla que lee proyecto_ml.ipynb si no tiene acceso a Drive
tesis/                        código Python, de las fotos a las métricas (su README lista cada módulo)
android/                      prototipo de verificación en el teléfono (su README explica la integración con ODK)
tests/                        pruebas unitarias: python -m pytest tests
```

Para revisar la tesis, el recorrido es: la [planificación](docs/planificacion_v4.md), los resultados en
[`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) y las [desviaciones](#desviaciones-respecto-de-la-planificación)
de abajo. El código está en [`tesis/`](tesis/) y [`android/`](android/).

## Estado por objetivo específico

| Objetivo (planificación 3.3) | Dónde | Estado |
|---|---|---|
| 1. Conjunto etiquetado y documentado | [`tesis/`](tesis/), pasos 1 y 2; EDA en [`proyecto_ml.ipynb`](notebooks/proyecto_ml.ipynb) | Hecho, con revisión manual en lugar de etiquetas débiles |
| 2. Partición por local, deduplicación, experimentos A y B | [`split.py`](tesis/split.py), [`build_dataset.py`](tesis/build_dataset.py) | Hecho |
| 3. Línea base y dos arquitecturas eficientes | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) | Hecho |
| 4. Cuantización y medición en el dispositivo | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) (tabla 5.7, frente de Pareto), [`quantize.py`](tesis/quantize.py), [`device.py`](tesis/device.py) | Hecho en el Honor X6b (A7) |
| 5. Prototipo de verificación local | [`android/`](android/) | Construido; falta correrlo en el Honor X6b (A8) |
| 6. Errores por clase y por cadena | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb), sección de análisis | Hecho |

## Resultados hasta ahora

Experimento A: prueba en locales no vistos, solo con fotos verificadas, sin las 92 fotos repetidas en otro local y con
18 capturas inválidas reales.

- **Arquitecturas.** El F1-macro en float32 es 0.955 para MobileNetV3-Large, 0.951 para EfficientNet-B0 y 0.950 para
  ResNet-50. La línea base no supera a la mejor eficiente (hipótesis 5.8): la diferencia es −0.005, con IC 95 %
  [−0.015, +0.015] por bootstrap sobre locales. ResNet-50 pesa 23 MB en su formato más chico, sobre el límite de 20 MB.
- **Cuantización.** float16 e int8 de rango dinámico pierden a lo sumo 0.002 de F1. El int8 completo rompe las redes
  eficientes: MobileNetV3 cae a 0.574 y EfficientNet a 0.792. ResNet-50 lo resiste.
- **En el teléfono** (Honor X6b, el modelo solo), los 17 formatos cumplen la latencia y la memoria. El peor es
  ResNet-50 en float32, con p95 de 246 ms. MobileNetV3 en float16 da p95 de 25 ms con 38 MB.
- **Verificación contra convenio** (MobileNetV3). Hay 2.1 % de falsas alarmas sobre pares por intercambio, y se
  detectan 84 de los 95 errores reales de campo registrados en ODK.
- **Viabilidad.** Ninguna configuración es viable todavía, y solo por el recall de rechazo: 0.889 (16 de 18) frente
  al umbral de 0.90. El candidato es MobileNetV3-Large; en float16 empata con float32 y pesa la mitad.

Pendiente:

- Reentrenar con las fotos mal tomadas y nuevas capturas inválidas reales, para el recall de rechazo.
- Correr el prototipo en el Honor X6b: latencia de extremo a extremo, paridad con la evaluación, aumento del paquete y
  prueba desde un formulario de ODK.
- Revisar las fotos de SOS y Promociones que entrenan sin verificar.
- Volver a correr `proyecto_ml.ipynb` sin las 92 fotos repetidas.

## Desviaciones respecto de la planificación

La planificación se sigue en lo siguiente:

- la taxonomía de 7 clases;
- la partición agrupada por local (las fotos de ODK conservan la asignación de su local);
- los experimentos A y B;
- la prueba solo con fotos verificadas;
- las tres arquitecturas con protocolo común;
- el int8 con calibración;
- la máscara con su control sin máscara;
- las métricas y umbrales de la tabla 5.7 ([`tesis/config.py`](tesis/config.py)).

Las desviaciones, que la memoria debe declarar, son estas:

- **Etiquetado (5.3, objetivo 1).** Se revisó a mano foto por foto, en lugar de usar etiquetas débiles de la maestra
  de convenios con una estimación de su ruido. No hay reconocimiento óptico de la superposición ni reconstrucción de
  la visita. Las 4 198 fotos de SOS y Promociones de ODK entrenan sin revisión (`verified=0`, fuera de la prueba),
  aunque la sección 5.1 pide verificarlas antes.
- **Taxonomía (2.2).** ARETE pasa a `lateral`; CHECK OUT | LATERAL, a `check_out`; MUEBLE GRIS, FOUR WAY y BIOMBO,
  a `exhibidor_piso`. OTA y el counter de las cadenas modernas pasan a `contenido_ajeno`, en lugar de quedar fuera de
  alcance.
- **Convenios (5.6, 5.9).** El proyecto no tiene la maestra. El convenio de cada local se aproxima con los muebles de
  sus fotos verificadas y los declarados en ODK (`tesis.metrics.contracts`). Es un superconjunto, así que las falsas
  alarmas y la exactitud condicionada son conservadoras.
- **Deduplicación (5.4).** Se quitan los cuasi-duplicados dentro del local (dhash ≤ 4) en lugar de reconstruir la
  visita. Entre locales solo se quita la misma foto (hash idéntico). Esto se agregó al encontrar 92 fotos de prueba
  repetidas en otro local, 88 de ellas en desarrollo. Los runs ya entrenados las excluyen al evaluar.
- **Aumentos (5.5).** El cambio de perspectiva se aproxima con zoom y traslación.
- **Máscara (5.5).** Cubre 0.70 × 0.33 ≈ 23 % de la imagen.
- **Tope por local.** `MAX_PER_STORE = 300` no se calibró.
- **Cuantización (5.5).** Además del int8 obligatorio, se prueban float16, int8 de rango dinámico e int8 con
  calibración balanceada.
- **Verificación (5.9).** ODK ya acumula errores reales de campo, así que se adelantó la validación con discrepancias
  reales, que la planificación dejaba como trabajo futuro.
- **Capturas inválidas reales (5.3).** La prueba A tiene 18:
  - dos de la revisión;
  - 16 fotos oscuras de ODK de locales de prueba, aprobadas después de entrenar (`entrenar_tesis` las predice con
    cada run y las suma).

  Las 14 del formulario *Fotos mal tomadas* son de locales de entrenamiento. Según 2.2 son 7 capturas inválidas y 7 de
  contenido ajeno; otras dos, de farmacias, quedan fuera de alcance. Entran en el próximo reentrenamiento y, mientras
  tanto, sirven de comprobación de campo.
- **Latencia (5.6).** `tesis.device` mide con `benchmark_model` de TFLite el modelo solo, con entrada aleatoria. La
  decodificación y el redimensionamiento de la foto, y el aumento del paquete, se miden con el prototipo. Cada ráfaga
  de 30 corre en un proceso nuevo y la herramienta da el p95 de cada ráfaga. Se reporta el mayor de las cuatro, que
  acota por arriba el p95 de las 120 ejecuciones.

## Decisiones de diseño que importan

- **Todo se valida agrupando por local.** El mismo local se fotografía cada mes; separar por foto mezclaría el mismo
  mueble en entrenamiento y prueba.
- **La superposición de Timestamp Camera se enmascara** dentro del modelo (0.70 × 0.33 abajo a la derecha, medido en
  777 fotos). Su última línea es el nombre del local, que delata la clase a través del convenio.
- **El 84 % de las fotos viene acostada**, con una etiqueta EXIF que TensorFlow ignora; `tesis.export` las endereza.
- **Las capturas inválidas sintéticas heredan el local de su origen**, así que no hay fuga, y nunca entran a la prueba.

## Reproducir

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
python -m pytest tests
```

- **Datos.** No se versionan. Los pasos para reconstruirlos están en el [README de `tesis/`](tesis/README.md#datos).
- **Entrenamiento**, en Colab: TensorFlow no usa la GPU en Windows nativo.
  1. Se sube `tesis_colab.zip` (lo genera `python -m tesis.export`) a `MyDrive/tesis`.
  2. Se abre el cuaderno con *Entorno de ejecución → GPU* y *Ejecutar todas*.

  `proyecto_ml.ipynb` tarda ≈ 1.5 h en una A100. `entrenar_tesis.ipynb` deja cada run en `MyDrive/tesis/runs_odk`.
- **Teléfono.** `python -m tesis.device runs/odk` mide los modelos con el teléfono por USB; los pasos están en su
  docstring. Para el prototipo, ver el [README de `android/`](android/README.md).

## Uso de IA

El código se escribió con asistencia de Claude Code (Anthropic). Los prompts que originaron cada parte están como
comentarios `# Prompt (Claude Code): ...` en las celdas de `proyecto_ml.ipynb`. Las decisiones de diseño se justifican
en los cuadernos y en los docstrings de cada módulo.
