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

Mientras corre la corrida final, `entrenar_tesis.ipynb` está sin salidas; las de la corrida anterior quedan en el
commit `b0e7b68`.

## Estado por objetivo específico

| Objetivo (planificación 3.3) | Dónde | Estado |
|---|---|---|
| 1. Conjunto etiquetado y documentado | [`tesis/`](tesis/), pasos 1 y 2; EDA en [`proyecto_ml.ipynb`](notebooks/proyecto_ml.ipynb) | Hecho, con revisión manual en lugar de etiquetas débiles |
| 2. Partición por local, deduplicación, experimentos A y B | [`split.py`](tesis/split.py), [`build_dataset.py`](tesis/build_dataset.py) | Hecho |
| 3. Línea base y dos arquitecturas eficientes | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) | Hecho |
| 4. Cuantización y medición en el dispositivo | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) (tabla 5.7, frente de Pareto), [`quantize.py`](tesis/quantize.py), [`device.py`](tesis/device.py) | Hecho en el Honor X6b (A7) |
| 5. Prototipo de verificación local | [`android/`](android/) | Hecho en el Honor X6b (A8), solo y desde una copia del formulario de ODK |
| 6. Errores por clase y por cadena | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb), sección de análisis | Hecho |

## Corrida final: qué cambia tras la revisión del tutor

La corrida anterior (más abajo) se está repitiendo con estas correcciones. Los resultados nuevos reemplazarán a los de
la sección siguiente cuando termine el entrenamiento.

- **Partición congelada antes de entrenar.** El conjunto se deduplicó primero y los locales se sortearon después, una
  sola vez más, el 16 de noviembre de 2026. La lista de fotos de cada partición y experimento está en
  `data/particion.csv`, con huella `5b7d0dc955c87813fa692fd762643fbb1431a37539af30d46b6935bb6225e5b7`
  (`tesis.split.fingerprint`; el cuaderno la imprime antes de entrenar). `tesis.split` comprueba que ningún local ni
  ninguna foto esté en dos particiones y falla si algo cambiaría la validación o la prueba.
- **Validación y prueba solo con fotos verificadas a mano.** Las 4 163 fotos de SOS y Promociones que entrenaban con
  etiqueta automática se revisaron una por una. En el entrenamiento solo quedan sin revisión manual las 1 423 capturas
  inválidas sintéticas, cuya etiqueta es por construcción.
- **Capturas inválidas reales.** Son 276: 90 en la prueba, 16 en validación y 170 en entrenamiento. Su recall se
  reporta con el número de fotos y el intervalo de Wilson al 95 %.
- **Convenios desde las maestras de Industrias Ales** (junio de 2025 y noviembre de 2026), no desde las etiquetas
  ([`tesis/convenios.py`](tesis/convenios.py)).
- **Experimento B** comparado con el A sobre las mismas fotos y las mismas clases.
- **Auditoría ciega** de una muestra de la validación y la prueba por una segunda persona
  ([`tesis/auditoria.py`](tesis/auditoria.py)); el resultado está más abajo.
- **Paquete de evidencias** sin fotos, con un comando que recalcula los resultados desde las predicciones
  ([`tesis/evidencia.py`](tesis/evidencia.py)).

Fotos del experimento A en la partición congelada:

| | chiclera | lateral | check out | cabecera | exhibidor de piso | captura inválida | contenido ajeno |
|---|---|---|---|---|---|---|---|
| Entrenamiento | 5 355 | 2 016 | 19 340 | 3 831 | 4 665 | 1 586 | 3 966 |
| Validación | 786 | 300 | 3 418 | 663 | 725 | 16 | 618 |
| Prueba | 1 551 | 634 | 7 354 | 1 470 | 1 547 | 90 | 1 326 |

La prueba tiene 13 972 fotos y no las 6 000 verificadas que preveía la planificación porque no se usaron etiquetas
débiles: todas las fotos se revisaron a mano, así que la prueba son todas las fotos verificadas de sus 112 locales.

### Auditoría ciega de las etiquetas

El 16 de noviembre de 2026, con la partición ya congelada y antes de tener resultados, una segunda persona clasificó
sin ver la etiqueta 756 fotos de la validación y la prueba: 100 de cada mueble, 150 de contenido ajeno y las 106
capturas inválidas. La auditoría mide las etiquetas; no las cambió.

| Clase | Fotos | Coinciden | Concordancia | IC 95 % (Wilson) |
|---|---|---|---|---|
| chiclera | 100 | 97 | 0.970 | [0.915, 0.990] |
| lateral | 100 | 93 | 0.930 | [0.863, 0.966] |
| check out | 100 | 98 | 0.980 | [0.930, 0.994] |
| cabecera | 100 | 98 | 0.980 | [0.930, 0.994] |
| exhibidor de piso | 100 | 100 | 1.000 | [0.963, 1.000] |
| captura inválida | 106 | 98 | 0.925 | [0.858, 0.961] |
| contenido ajeno | 150 | 142 | 0.947 | [0.898, 0.973] |
| todas | 756 | 726 | 0.960 | [0.944, 0.972] |

El kappa de Cohen es 0.953. De los 30 desacuerdos, 13 son entre las dos categorías de rechazo y 6 son laterales que
la segunda persona vio como exhibidor de piso. Que la foto debe rechazarse casi no se discute: de las 256 fotos de
rechazo, la segunda persona rechaza 253 con una u otra categoría. Lo que se discute es cuál: 7 de las 90 capturas
inválidas de la prueba le parecieron contenido ajeno (acercamientos legibles a una etiqueta de precio, perchas con
medio piso, la entrada de un local).

Por eso, además del recall de cada rechazo que exige la tabla 5.7, los resultados llevarán dos complementos, definidos
aquí antes de tener resultados: el recall de captura inválida sobre las fotos en que las dos personas coinciden, y la
fracción de fotos de rechazo que el modelo rechaza con cualquiera de las dos categorías.

## Resultados de la corrida anterior (10 de noviembre de 2026)

Son los que revisó el tutor, con la partición del 26 de octubre y el convenio aproximado con las etiquetas.
Experimento A: prueba en locales no vistos, solo con fotos verificadas, sin las 92 fotos repetidas en otro local y con
18 capturas inválidas reales.

- **Arquitecturas.** El F1-macro en float32 es 0.955 para MobileNetV3-Large, 0.951 para EfficientNet-B0 y 0.950 para
  ResNet-50. La línea base no supera a la mejor eficiente (hipótesis 5.8): la diferencia es −0.005, con IC 95 %
  [−0.015, +0.015] por bootstrap sobre locales. ResNet-50 pesa 23 MB en su formato más chico, sobre el límite de 20 MB.
- **Cuantización.** float16 e int8 de rango dinámico pierden a lo sumo 0.002 de F1. El int8 completo rompe las redes
  eficientes: MobileNetV3 cae a 0.574 y EfficientNet a 0.792. ResNet-50 lo resiste.
- **En el teléfono** (Honor X6b, el modelo solo), los 17 formatos cumplen la latencia y la memoria. El peor es
  ResNet-50 en float32, con p95 de 246 ms. MobileNetV3 en float16 da p95 de 25 ms con 38 MB.
- **Prototipo, de extremo a extremo** (Honor X6b, MobileNetV3 en float16, 120 fotos reales de ODK). La mediana es
  67 ms y el p95, 118 ms, de los que 17 ms son leer y reducir la foto; la memoria pico es 172 MB. El modelo y el
  runtime suman 10.0 MB, el 55 % del APK de ODK Collect. El teléfono da la misma clase que la evaluación en las 120.
- **Desde ODK Collect.** En una copia del formulario de muebles, asignada a un solo usuario, la pregunta de foto abre
  el prototipo y este, Timestamp Camera. Una foto oscura de 2560 × 1920 se rechazó como captura inválida en 193 ms,
  volvió al formulario y llegó a Central como adjunto.
- **Verificación contra convenio** (MobileNetV3). Hay 2.1 % de falsas alarmas sobre pares por intercambio, y se
  detectan 84 de los 95 errores reales de campo registrados en ODK.
- **Viabilidad.** Ninguna configuración es viable todavía, y solo por el recall de rechazo: 0.889 (16 de 18) frente
  al umbral de 0.90. El candidato es MobileNetV3-Large; en float16 empata con float32 y pesa la mitad.

Pendiente:

- Entrenar la corrida final en Colab con la partición congelada (`entrenar_tesis.ipynb`).
- Medir en el Honor X6b los modelos nuevos, sin conexión, y repetir la comparación de predicciones entre el teléfono y
  la evaluación.
- Probar el prototipo desde ODK frente a un mueble real: la prueba del formulario fue con una foto oscura, que solo
  ejercita el rechazo.
- Investigar por qué el int8 completo degrada las redes eficientes (calibración, distribución de las imágenes,
  operaciones).
- Volver a correr `proyecto_ml.ipynb` con el conjunto deduplicado.

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
  la visita. La herramienta muestra páginas de 24 fotos con una clase sugerida y confirma las no tocadas, así que la
  bitácora registra cada decisión pero no prueba que cada foto se miró: por eso se agrega la auditoría ciega.
- **Taxonomía (2.2).** ARETE pasa a `lateral`; CHECK OUT | LATERAL, a `check_out`; MUEBLE GRIS, FOUR WAY y BIOMBO,
  a `exhibidor_piso`. OTA y el counter de las cadenas modernas pasan a `contenido_ajeno`, en lugar de quedar fuera de
  alcance.
- **Convenios (5.6, 5.9).** El convenio de cada local sale de dos maestras de muebles alquilados de Industrias Ales:
  la de junio de 2025 y el catálogo de noviembre de 2026. Cada una rige las fotos tomadas a menos de seis meses de su
  fecha; las demás quedan fuera de la verificación. Las maestras no traen los muebles ganados en negociación ni, casi,
  a Favorita, así que la verificación cubre menos de la mitad de la prueba (6 035 de 13 972 fotos). La maestra de 2025
  no usa el nombre chiclera: en Tía, su fila de check out de Mars se toma como check out más chiclera, porque así la
  desglosa el catálogo de 2026 en 117 de los 119 locales que están en las dos. La corrida anterior aproximaba el
  convenio con las etiquetas de las fotos.
- **Deduplicación (5.4).** Se quitan los cuasi-duplicados dentro del local (dhash ≤ 4) en lugar de reconstruir la
  visita. Entre locales se quita toda foto archivada en más de un local (hash idéntico; 672 fotos), porque no se sabe
  de cuál es. Esto se agregó al encontrar 92 fotos de prueba repetidas en otro local, y ahora ocurre antes del sorteo.
- **Partición (5.4).** Los locales se sortearon de nuevo el 16 de noviembre de 2026, después de deduplicar; 18 de los
  112 locales de prueba ya lo eran. Tres locales son de prueba por diseño y no por sorteo: Coral de la Y, Supermaxi
  Iñaquito y Santa María Iñaquito, donde el autor tomó a propósito las capturas inválidas para evaluar. Las capturas
  inválidas reales se estratifican aparte de las sintéticas.
- **Aumentos (5.5).** El cambio de perspectiva se aproxima con zoom y traslación.
- **Máscara (5.5).** Cubre 0.70 × 0.33 ≈ 23 % de la imagen.
- **Tope por local.** `MAX_PER_STORE = 300` no se calibró.
- **Cuantización (5.5).** Además del int8 obligatorio, se prueban float16, int8 de rango dinámico e int8 con
  calibración balanceada.
- **Verificación (5.9).** ODK ya acumula errores reales de campo, así que se adelantó la validación con discrepancias
  reales, que la planificación dejaba como trabajo futuro.
- **Capturas inválidas reales (5.3).** Los mercaderistas están entrenados para no tomarlas, así que casi no hay en el
  histórico (dos). Las 276 de la corrida final vienen de:
  - el formulario *Fotos mal tomadas* de ODK, donde los mercaderistas las tomaron a propósito (las de farmacias quedan
    fuera de alcance);
  - 64 fotos que el autor tomó a propósito el 16 de noviembre de 2026 en locales que por eso son de prueba;
  - las fotos de SOS y Promociones que la revisión manual encontró oscuras, tapadas, movidas o dirigidas al piso.

  Las que son nítidas y muestran otra cosa (una percha, un pasillo, una captura de pantalla) son contenido ajeno,
  como define la sección 2.2, aunque se hayan tomado como ejemplo de foto mala: 98 capturas de pantalla y 11 de las 64
  fotos del autor quedaron así. En la corrida anterior la prueba tenía 18 capturas inválidas, 16 de ellas sumadas
  después de entrenar, y una se reetiquetó después de evaluar; la partición congelada evita las dos cosas.
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
