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

`entrenar_tesis.ipynb` trae las salidas de la corrida final (17 de noviembre de 2026), ejecutada con el código del
commit `debea6f`; las de la corrida anterior quedan en el commit `b0e7b68`.

## Estado por objetivo específico

| Objetivo (planificación 3.3) | Dónde | Estado |
|---|---|---|
| 1. Conjunto etiquetado y documentado | [`tesis/`](tesis/), pasos 1 y 2; EDA en [`proyecto_ml.ipynb`](notebooks/proyecto_ml.ipynb) | Hecho, con revisión manual en lugar de etiquetas débiles |
| 2. Partición por local, deduplicación, experimentos A y B | [`split.py`](tesis/split.py), [`build_dataset.py`](tesis/build_dataset.py) | Hecho |
| 3. Línea base y dos arquitecturas eficientes | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) | Hecho |
| 4. Cuantización y medición en el dispositivo | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb) (tabla 5.7, frente de Pareto), [`quantize.py`](tesis/quantize.py), [`device.py`](tesis/device.py) | Hecho en el Honor X6b (A7), sin conexión, con los modelos de la corrida final |
| 5. Prototipo de verificación local | [`android/`](android/) | Hecho en el Honor X6b (A8) con el modelo de la corrida final. La prueba desde una copia del formulario de ODK es con el modelo anterior |
| 6. Errores por clase y por cadena | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb), sección de análisis | Hecho |

## Corrida final: qué cambia tras la revisión del tutor

La corrida que revisó el tutor se repitió con estas correcciones. Los resultados están en la
[sección siguiente](#resultados-de-la-corrida-final-17-de-noviembre-de-2026).

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

Por eso, además del recall de cada rechazo que exige la tabla 5.7, los resultados llevan dos complementos, definidos
aquí antes de tener resultados (commit `debea6f`): el recall de captura inválida sobre las fotos en que las dos
personas coinciden, y la fracción de fotos de rechazo que el modelo rechaza con cualquiera de las dos categorías.

## Resultados de la corrida final (17 de noviembre de 2026)

Experimento A: 13 972 fotos de prueba de 112 locales que el modelo no vio, todas verificadas a mano. Antes de calcular
nada, el cuaderno comprueba que cada run se evaluó con la prueba congelada.

| Arquitectura (float32) | MB | F1-macro | Recall macro | Recall de captura inválida (90 fotos) | Recall de contenido ajeno (1 326 fotos) |
|---|---|---|---|---|---|
| MobileNetV3-Large | 12.4 | 0.920 | 0.903 | 0.644 [0.541, 0.736] | 0.965 [0.953, 0.973] |
| EfficientNet-B0 | 16.6 | 0.919 | 0.905 | 0.533 [0.431, 0.633] | 0.974 [0.964, 0.981] |
| ResNet-50 | 91.0 | 0.903 | 0.884 | 0.489 [0.388, 0.590] | 0.950 [0.937, 0.961] |
| MobileNetV3-Large sin máscara (control) | 12.1 | 0.929 | 0.911 | 0.578 [0.475, 0.675] | 0.983 [0.975, 0.989] |

Entre corchetes va el intervalo de Wilson al 95 %. Los umbrales de la tabla 5.7 son F1-macro ≥ 0.85, recall macro
≥ 0.90 y recall de cada rechazo ≥ 0.90.

- **Viabilidad.** Ninguna configuración es viable, porque ninguna arquitectura alcanza el recall de captura inválida.
  La mejor es MobileNetV3-Large, con 58 de 90. MobileNetV3 y EfficientNet-B0, en float32 y en float16, cumplen todos
  los demás criterios de la tabla 5.7, incluidos los del teléfono.
- **Capturas inválidas.** De las 90, MobileNetV3 da 58 como captura inválida y 24 como contenido ajeno. Rechaza
  entonces 82 con una u otra categoría (0.911, IC 95 % [0.834, 0.954]) y acepta 8 como mueble: 5 como exhibidor de
  piso, 2 como check out y 1 como chiclera. El fallo es parecido en las 53 fotos que tomó el autor (reconoce 36) y en
  las 36 que vienen de los formularios de ODK (22). Los dos complementos de la auditoría:
  - En las 83 capturas inválidas en que las dos personas coinciden, el recall es 0.675 (56 de 83, [0.568, 0.766]).
    El desacuerdo entre personas no explica la distancia al umbral.
  - De las 1 416 fotos de rechazo de la prueba, el modelo rechaza 1 363 con una u otra categoría (0.963,
    [0.951, 0.971]).

  A la vista, las 32 que no reconoce son casi todas fotos movidas o inclinadas en las que todavía se distingue una
  percha o un mueble, y encuadres al piso o al techo con algo de percha. El modelo responde a lo que se ve y no a la
  calidad de la toma.
- **Arquitecturas (hipótesis 5.8).** ResNet-50 no supera a MobileNetV3: la diferencia de F1-macro es −0.017, con
  IC 95 % [−0.039, +0.011] por bootstrap sobre locales. Además incumple el recall macro y pesa 23 MB en su formato más
  chico, sobre el límite de 20 MB.
- **Por clase** (MobileNetV3). El recall más bajo entre los muebles es el de lateral, 0.844 [0.814, 0.870]: uno de
  cada diez laterales sale como exhibidor de piso. Los otros cuatro están entre 0.918 y 0.989.
- **Cuantización.** float16 no pierde F1 frente a float32 (coinciden en más del 99.9 % de las fotos) y pesa la mitad:
  5.9 MB en MobileNetV3. El int8 de rango dinámico pierde a lo sumo 0.002 de F1. El int8 completo, el que pide el
  objetivo, rompe las redes eficientes: MobileNetV3 cae a 0.594 (caída de 0.325, IC 95 % [0.260, 0.378]) y
  EfficientNet-B0 a 0.839 (0.081, [0.061, 0.106]). ResNet-50 lo resiste (0.902). Calibrar con las clases balanceadas
  no lo arregla: 0.607 y 0.822.
- **Verificación contra el convenio** (MobileNetV3, con el convenio de las maestras). De la prueba, 5 037 fotos son
  de un local con convenio fechado y muestran un mueble de ese convenio o un rechazo. En ellas la exactitud es 0.980,
  y 0.986 si la predicción se restringe a los muebles del convenio.
  - Pares por intercambio en el histórico (3 333 correctos y 5 515 discrepantes): 1.3 % de falsas alarmas y 99.3 % de
    detección con la predicción restringida; 1.6 % y 99.3 % sin restringir.
  - Lo que el mercaderista declaró en ODK, cuando ese mueble está en el convenio: 0.4 % de falsas alarmas en 724 fotos
    correctas y 17 de 17 errores detectados.
  - Todas las fotos de ODK con mueble declarado, sin restringir: 3.0 % de falsas alarmas en 1 075 fotos correctas y
    65 de 73 errores detectados.

  El umbral de falsas alarmas es 5 %.
- **Por cadena** (MobileNetV3). La exactitud va de 0.891 (Favorita, histórico) a 0.985 (Tía, ODK), sin contar las 7
  fotos de Liris. Los errores más frecuentes son lateral por exhibidor de piso (Tía y Santa María), exhibidor de piso
  por cabecera (Favorita) y check out por cabecera (Coral).

La corrida anterior daba a MobileNetV3 un F1-macro de 0.955 y 16 de 18 capturas inválidas. La prueba es otra (locales
sorteados de nuevo y 90 capturas inválidas reales en lugar de 18), así que las dos cifras no se comparan foto a foto,
y no se midió cuánto aporta cada cambio.

### Experimento B (temporal)

La prueba de B son las 1 329 fotos del histórico tomadas desde octubre de 2025 en los locales de prueba: 332
chicleras, 88 laterales, 665 check out, 122 cabeceras, 119 exhibidores de piso y 3 de contenido ajeno. No tiene
capturas inválidas, así que el recall de ese rechazo no se puede medir en B. El F1-macro promedia las seis clases
presentes.

| Arquitectura | F1 de B en su prueba | F1 de A en las mismas fotos | A − B, IC 95 % | F1 de B antes del corte |
|---|---|---|---|---|
| ResNet-50 | 0.779 | 0.790 | +0.010 [−0.008, +0.030] | 0.949 |
| EfficientNet-B0 | 0.846 | 0.836 | −0.010 [−0.042, +0.019] | 0.949 |
| MobileNetV3-Large | 0.805 | 0.782 | −0.023 [−0.040, −0.008] | 0.954 |

«Antes del corte» son las 10 166 fotos del histórico anteriores a octubre de 2025, de los mismos locales y las mismas
clases.

- El modelo A entrenó con fotos de todo el periodo y no rinde mejor que el B en estas fotos. La baja frente al periodo
  anterior no viene, entonces, de haber entrenado solo con fotos antiguas.
- Casi toda la baja es aritmética. Contenido ajeno tiene 3 fotos y pesa un sexto del promedio; MobileNetV3 y
  ResNet-50 no aciertan ninguna, y EfficientNet-B0, una. Sobre las cinco clases de mueble, el F1-macro de B es 0.966
  en MobileNetV3, 0.958 en EfficientNet-B0 y 0.935 en ResNet-50; el de A en las mismas fotos es 0.939, 0.966 y 0.948.

### Por qué falla el int8

`python -m tesis.quantize <run> --diagnostico` repite la cuantización con variantes y mide cada una sobre 280 fotos de
la prueba, 40 por clase. La tabla da la exactitud. El resultado es negativo: ninguna variante recupera el float32.

| Variante | MobileNetV3-Large | EfficientNet-B0 |
|---|---|---|
| float32 | 0.911 | 0.914 |
| int8 del protocolo, calibrado con 300 fotos | 0.593 | 0.836 |
| int8 calibrado con 1 000 fotos | 0.582 | 0.861 |
| int8 con un tipo de operación en float (el mejor de los probados) | 0.650 | 0.864 |
| int8 con el 10 % de los tensores de más error en float | 0.614 | 0.814 |

- **No es la calibración.** Ni más fotos ni clases balanceadas (arriba, sobre toda la prueba) lo arreglan.
- **No es una operación.** Se dejó en float cada tipo de operación, uno a la vez. El mejor es la convolución en
  profundidad en MobileNetV3 y la multiplicación en EfficientNet-B0. Con 280 fotos, diferencias de 0.03 entre
  variantes no son significativas.
- **No es una capa.** El error de cada operación por separado es chico: la mediana es 0.29 pasos de cuantización en
  las dos redes, y el máximo, 2.0 en MobileNetV3 y 9.7 en EfficientNet-B0.
- **Lo que sí se ve** son activaciones de rango muy amplio: hasta 558 en MobileNetV3 y 440 en EfficientNet-B0, donde
  cada uno de los 256 niveles vale 2.2 y 1.7. Es compatible con una pérdida repartida por toda la red, pero el
  diagnóstico no lo prueba.

El detalle está en `int8_diagnostico.csv` e `int8_capas.csv` de cada run, dentro del paquete de evidencias. El formato
para el teléfono es float16, que no pierde exactitud; queda aparte del objetivo de int8, que no se cumple en las redes
eficientes.

### En el teléfono

Medido el 17 de noviembre de 2026 en el Honor X6b con los modelos de la corrida final: en modo avión y sin red
activa, con ODK Collect abierto y la pantalla encendida. Cada ráfaga deja registradas esas condiciones. La tabla
5.7 y el frente de Pareto del cuaderno se volvieron a ejecutar con estas mediciones; sus demás salidas son las de
Colab, anteriores a la medición.

- **El modelo solo** (los 17 formatos, 4 ráfagas de 30 inferencias con `benchmark_model`). Todos cumplen la latencia
  (p50 ≤ 2 000 ms y p95 ≤ 3 000 ms) y la memoria (≤ 300 MB). El más lento es ResNet-50 en float16, con p95 de 230 ms,
  y el que más memoria usa, ResNet-50 en float32, con 220 MB. MobileNetV3 en float16 da p50 de 19 ms y p95 de 25 ms
  con 39 MB. El int8 de MobileNetV3 es más lento que su float32 (p50 de 38 ms frente a 19 ms): XNNPACK no lo acepta y
  corre sin él. En el frente de Pareto de F1-macro frente a p95 quedan MobileNetV3 en float32 y en float16 y
  EfficientNet-B0 en int8 de rango dinámico, los tres con F1-macro de 0.92.
- **Prototipo, de extremo a extremo** (MobileNetV3 en float16, 4 ráfagas de 30 fotos).

  | Fotos | p50 | p95 | Leer y reducir la foto | Memoria pico | Misma clase que la evaluación |
  |---|---|---|---|---|---|
  | 120 de ODK de locales de prueba, como las guarda ODK (≈1024 px) | 71 ms | 110 ms | 18 ms | 175 MB | 120 de 120 |
  | 53 capturas inválidas a resolución completa, como salen de la cámara | 108 ms | 159 ms | 54 ms | 233 MB | 51 de 53 |

  El modelo y el runtime suman 10.0 MB, el 55 % del APK de ODK Collect. El teléfono y la evaluación reducen la foto
  con bibliotecas distintas, y la diferencia crece con el tamaño de la foto: la mayor diferencia de probabilidad por
  foto tiene mediana 0.000 en las de ODK y 0.024 en las de cámara. En las dos fotos de cámara en que no coinciden,
  ninguno de los dos dice captura inválida; en una, el teléfono acepta un exhibidor de piso donde la evaluación
  rechaza por contenido ajeno.
- **Desde ODK Collect** (15 de noviembre, con el modelo de la corrida anterior). En una copia del formulario de
  muebles, asignada a un solo usuario, la pregunta de foto abre el prototipo y este, Timestamp Camera. Una foto oscura
  de 2560 × 1920 se rechazó como captura inválida en 193 ms, volvió al formulario y llegó a Central como adjunto.

### Pendiente

- Probar el prototipo desde ODK frente a un mueble real: la prueba del formulario fue con una foto oscura, que solo
  ejercita el rechazo.

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

  `proyecto_ml.ipynb` tarda ≈ 1.5 h en una A100. Su última corrida, del 17 de noviembre de 2026 con la partición
  congelada, fue local y sin GPU: 1 h para los rasgos de las dos redes y 2 h 40 min para el resto. En Windows con
  poca memoria libre hay que correrlo con `$env:LOKY_MAX_CPU_COUNT = 1`: en paralelo copia los datos a cada proceso y
  falla. `entrenar_tesis.ipynb` tardó 5.3 h de entrenamiento y 47 min de
  cuantización en una L4 con RAM alta; deja cada run en `MyDrive/tesis/runs_final` y el paquete de evidencias en
  `MyDrive/tesis/evidencia_tesis.zip`.
- **Teléfono.** `python -m tesis.device runs/final` mide los modelos con el teléfono por USB; los pasos están en su
  docstring. Para el prototipo, ver el [README de `android/`](android/README.md).

## Uso de IA

El código se escribió con asistencia de Claude Code (Anthropic). Los prompts que originaron cada parte están como
comentarios `# Prompt (Claude Code): ...` en las celdas de `proyecto_ml.ipynb`. Las decisiones de diseño se justifican
en los cuadernos y en los docstrings de cada módulo.
