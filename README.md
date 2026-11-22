# Clasificación de muebles de exhibición en dispositivos de gama baja

Soy Sebastián Garzón. Este repositorio tiene mi tesis de la Maestría en Ciencia de Datos (USFQ) y el proyecto de
Machine Learning del curso que se desprende de ella. Tutor: Ricardo Flores Moyano.

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
| 5. Prototipo de verificación local | [`android/`](android/) | Hecho en el Honor X6b (A8) con el modelo de la corrida final, también desde una copia del formulario de ODK en un supermercado. La decisión corregida contra el convenio está instalada en el teléfono desde el 22 de noviembre; falta probarla en un local |
| 6. Errores por clase y por cadena | [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb), sección de análisis | Hecho |

## Corrida final: qué cambia respecto de la anterior

Repetí la corrida anterior con estas correcciones. Los resultados están en la
[sección siguiente](#resultados-de-la-corrida-final-17-de-noviembre-de-2026).

- **Partición congelada antes de entrenar.** El conjunto se deduplicó primero y los locales se sortearon después, una
  sola vez más, el 16 de noviembre de 2026. La lista de fotos de cada partición y experimento está en
  `data/particion.csv`, con huella `5b7d0dc955c87813fa692fd762643fbb1431a37539af30d46b6935bb6225e5b7`
  (`tesis.split.fingerprint`; el cuaderno la imprime antes de entrenar). `tesis.split` comprueba que ningún local ni
  ninguna foto esté en dos particiones y falla si algo cambiaría la validación o la prueba.
- **Validación y prueba solo con fotos verificadas a mano.** Las 4 163 fotos de SOS y Promociones que entrenaban con
  etiqueta automática se revisaron una por una. En el entrenamiento solo quedan sin revisión manual las 1 423 capturas
  inválidas sintéticas, cuya etiqueta es por construcción.
- **Capturas inválidas reales.** El corpus tiene 276: 90 en locales de prueba, 16 en locales de validación y 170 en
  locales de entrenamiento. El entrenamiento usa 163 de esas 170; las otras 7 quedaron fuera por el tope de fotos por
  local (ver la tabla de abajo). Su recall se reporta con el número de fotos y el intervalo de Wilson al 95 %, y
  separando las que se tomaron mal a propósito de las espontáneas.
- **Convenios desde las maestras de Industrias Ales** (junio de 2025 y noviembre de 2026), no desde las etiquetas
  ([`tesis/convenios.py`](tesis/convenios.py)).
- **Experimento B** comparado con el A sobre las mismas fotos y las mismas clases.
- **Auditoría ciega** de una muestra de la validación y la prueba por una segunda persona
  ([`tesis/auditoria.py`](tesis/auditoria.py)); el resultado está más abajo.
- **Paquete de evidencias** sin fotos, en dos tamaños: el completo, de 739 MB, con los modelos entrenados, y el
  reducido, de 19 MB, sin ellos. Los dos traen un comando que recalcula los resultados desde las predicciones y los
  compara con los reportados ([`tesis/evidencia.py`](tesis/evidencia.py); ver
  [cómo comprobarlos](#reproducir)).

Fotos del experimento A en la partición congelada:

| | chiclera | lateral | check out | cabecera | exhibidor de piso | captura inválida | contenido ajeno |
|---|---|---|---|---|---|---|---|
| Entrenamiento | 5 355 | 2 016 | 19 340 | 3 831 | 4 665 | 1 586 | 3 966 |
| Validación | 786 | 300 | 3 418 | 663 | 725 | 16 | 618 |
| Prueba | 1 551 | 634 | 7 354 | 1 470 | 1 547 | 90 | 1 326 |

La tabla cuenta las fotos que cada partición usa, que en entrenamiento no son todo el corpus. Los locales de
entrenamiento tienen 47 118 fotos y el experimento A entrena con 40 759: el tope de 300 fotos por local
(`MAX_PER_STORE`) deja fuera, al azar, 6 359 fotos de los 37 locales que lo superan. Entre ellas hay 7 capturas
inválidas reales. Las 1 586 capturas inválidas de entrenamiento son, entonces, 163 reales y 1 423 sintéticas. En
validación y prueba no hay tope ni sintéticas.

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

Los 30 desacuerdos no se arbitraron ni se corrigieron. Son 25 fotos de la prueba y 5 de la validación, y las dos
particiones estaban congeladas cuando se hizo la auditoría: cambiar una etiqueta de la prueba después de congelarla es
lo que la congelación impide. Quedan listados foto por foto, con las dos clases, en `datos/auditoria.csv` del paquete
de evidencias. De los 25 de la prueba, 12 son entre las dos categorías de rechazo, 8 entre muebles y 5 sobre si la
foto se rechaza o no.

En lugar de corregirlos, se mide cuánto pesan. Además del recall de cada rechazo que exige la tabla 5.7, los
resultados llevan dos complementos, definidos aquí antes de tener resultados (commit `debea6f`): el recall de captura
inválida sobre las fotos en que las dos personas coinciden, y la fracción de fotos de rechazo que el modelo rechaza
con cualquiera de las dos categorías. Después de la corrida se agregó la lectura contraria: los mismos recalls con la
clase que dio la segunda persona ([más abajo](#cuánto-pesan-los-desacuerdos-de-la-auditoría)).

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
  piso, 2 como check out y 1 como chiclera. De las 90, 62 se tomaron mal a propósito y 28 son espontáneas; el
  recall de cada grupo está [más abajo](#capturas-inválidas-inducidas-y-espontáneas). Los dos complementos de la
  auditoría:
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
- **Verificación contra el convenio** (MobileNetV3). Con la decisión corregida, que ya no restringe a ciegas al
  convenio, las falsas alarmas son 1.5 % en los pares por intercambio (el límite es 5 %) y 3.0 % en las declaraciones
  reales de ODK, donde se detectan 67 de 73 errores de campo. El detalle, la cobertura y la diferencia entre los dos
  experimentos están [más abajo](#verificación-contra-el-convenio).
- **Por cadena** (MobileNetV3). La exactitud va de 0.891 (Favorita, histórico) a 0.985 (Tía, ODK), sin contar las 7
  fotos de Liris. Los errores más frecuentes son lateral por exhibidor de piso (Tía y Santa María), exhibidor de piso
  por cabecera (Favorita) y check out por cabecera (Coral).

La corrida anterior daba a MobileNetV3 un F1-macro de 0.955 y 16 de 18 capturas inválidas. La prueba es otra (locales
sorteados de nuevo y 90 capturas inválidas reales en lugar de 18), así que las dos cifras no se comparan foto a foto,
y no se midió cuánto aporta cada cambio.

### Capturas inválidas inducidas y espontáneas

De las 90 capturas inválidas reales de la prueba, 62 se tomaron mal a propósito, para tener con qué evaluar, y 28
ocurrieron solas durante una visita. Con MobileNetV3 en float32:

| Origen | Fotos | Locales | Reconocidas como captura inválida | Rechazadas con una u otra categoría |
|---|---|---|---|---|
| Inducidas: las tomé yo el 16 de noviembre de 2026 | 53 | 2 | 36 (0.679 [0.545, 0.789]) | 48 (0.906 [0.797, 0.959]) |
| Inducidas: mercaderistas, en el formulario *Fotos mal tomadas* | 9 | 3 | 2 (0.222 [0.063, 0.547]) | 8 (0.889 [0.565, 0.980]) |
| Espontáneas: halladas al revisar SOS, Promociones, el formulario de muebles y el histórico | 28 | 11 | 20 (0.714 [0.529, 0.847]) | 26 (0.929 [0.774, 0.980]) |
| Todas | 90 | 16 | 58 (0.644 [0.541, 0.736]) | 82 (0.911 [0.834, 0.954]) |

- **Las inducidas no son errores de una visita.** Las mías las tomé en un solo día, en dos locales que por eso son de
  prueba por diseño y no por sorteo. Las de los mercaderistas salen de un sorteo en que se les pidió tomar fotos
  malas. Quien toma mal una foto a propósito elige cómo hacerlo, y puede exagerar el defecto o usar encuadres que en
  una visita no ocurren.
- **El entrenamiento es al revés.** De las 163 capturas inválidas reales con que se entrena, 156 son espontáneas y 7
  son inducidas, todas de mercaderistas. Las 16 de validación son espontáneas. En el corpus completo son 206
  espontáneas y 70 inducidas (53 mías y 17 de mercaderistas).
- **Cómo leer estas cifras.** No estiman cuántas capturas inválidas hay en producción ni el desempeño que el modelo
  tendría allí: la mezcla de la prueba la decidí yo, no una muestra de visitas. Lo que sostienen es la conclusión de
  viabilidad, porque el criterio (≥ 0.90) no se alcanza con ningún origen. En las espontáneas, que son las que se
  parecen a producción, el recall es 0.714 y su intervalo llega a 0.847; con 28 fotos de 11 locales no se puede
  afirmar más.

### Cuánto pesan los desacuerdos de la auditoría

Los recalls de rechazo de MobileNetV3 sobre las fotos auditadas de la prueba, leídos de tres maneras:

| | Con la etiqueta | Solo donde las dos personas coinciden | Con la clase de la segunda persona |
|---|---|---|---|
| Recall de captura inválida | 58 de 90 = 0.644 [0.541, 0.736] | 56 de 83 = 0.675 [0.568, 0.766] | 56 de 89 = 0.629 [0.525, 0.722] |
| Recall de contenido ajeno (muestra auditada) | 98 de 103 = 0.951 [0.891, 0.979] | 93 de 95 = 0.979 [0.926, 0.994] | 98 de 103 = 0.951 [0.891, 0.979] |

- **Captura inválida** queda bajo el umbral de 0.90 con las tres lecturas; la más favorable llega a 0.766 en el
  extremo de su intervalo. El resultado no se explica por errores de etiquetado.
- **Contenido ajeno** da lo mismo con la etiqueta que con la clase de la segunda persona. Sobre toda la prueba (1 326
  fotos) su recall es 0.965 [0.953, 0.973].
- **En los 25 desacuerdos de la prueba**, el modelo coincide con la etiqueta en 14, con la segunda persona en 8 y con
  ninguna en 3.
- **Alcance.** Las capturas inválidas se auditaron todas; de las demás clases, una muestra. En el resto de la prueba
  cabe esperar una proporción parecida de etiquetas discutibles, cerca del 4 %, que la auditoría no localiza.

### Verificación contra el convenio

**Cobertura.** Las maestras de Industrias Ales solo alcanzan una parte de la prueba. De las 13 972 fotos, 6 035
(43 %) son de un local con convenio fechado: 61 de los 112 locales. De esas, 5 037 muestran un mueble de ese convenio
o un rechazo, y 998 muestran un mueble que el convenio no lista, porque las maestras no traen los ganados en
negociación. La exactitud condicionada de la planificación se mide sobre las 5 037 (el 36 % de la prueba): 0.980 sin
restringir y 0.986 con la predicción restringida al convenio. El resto de la prueba no tiene con qué verificarse.

**La restricción convertía discrepancias en coincidencias.** Restringida al convenio, la predicción solo puede ser
un mueble contratado o un rechazo. Si la foto muestra otro mueble, la restricción lo hace pasar por uno contratado.
Medido en la prueba con MobileNetV3:

- En los 21 locales con un solo mueble en el convenio hay 333 fotos que muestran otro mueble. Restringida, la
  decisión da el mueble contratado en 172 (52 %): si ese era el declarado, no hay alerta.
- En las 998 fotos de un mueble que el convenio no lista, la decisión restringida no acierta nunca, por construcción.
  Sobre las 6 035 fotos con convenio fechado su exactitud es 0.823, frente a 0.974 sin restringir.

**La decisión corregida** (`tesis.metrics.decide`, y `Clasificador.decidir` en el prototipo). La clase más probable
se conserva cuando es un rechazo, un mueble del convenio o un mueble que el convenio no lista con probabilidad ≥ τ:
hay evidencia de que la foto muestra otro mueble. Solo si esa evidencia no alcanza se decide dentro del convenio.
τ se calibra con la validación de cada red (`calibrar_umbral`): es el que maximiza la detección de un mueble fuera
del convenio menos las falsas alarmas. La prueba no interviene. Sale 0.47 para MobileNetV3, 0.56 para EfficientNet-B0
y 0.50 para ResNet-50.

| MobileNetV3, en la prueba | Restringida al convenio | Sin restringir | Con umbral (τ = 0.47) |
|---|---|---|---|
| Exactitud en las 6 035 fotos con convenio fechado | 0.823 | 0.974 | 0.974 |
| Locales con un solo mueble: alerta cuando la foto muestra otro (333) | 161 (48 %) | 333 (100 %) | 333 (100 %) |
| Locales con un solo mueble: falsas alarmas (243 fotos del contratado) | 3 (1.2 %) | 13 (5.3 %) | 13 (5.3 %) |
| Pares por intercambio: falsas alarmas (3 333 correctos) | 1.3 % | 1.6 % | 1.5 % |
| Pares por intercambio: detección (5 515 discrepantes) | 99.3 % | 99.3 % | 99.3 % |
| Pares con un mueble fuera del convenio: detección (1 261) | 56.1 % | 98.8 % | 98.7 % |
| ODK, declaraciones reales: falsas alarmas (1 075 correctas) | 27.4 % | 3.0 % | 3.0 % |
| ODK, declaraciones reales: errores detectados (73) | 72 | 65 | 67 |

- **Lo que se gana.** El caso de un mueble fuera del convenio pasa de detectarse en 56 % de los pares a 99 %, y en
  los locales con un solo mueble ya no pasa ninguno por el contratado.
- **Lo que cuesta.** Las falsas alarmas suben de 1.3 % a 1.5 % en los pares por intercambio, bajo el límite de 5 % de
  la tabla 5.7. En los locales con un solo mueble suben de 3 a 13 fotos de 243 (5.3 %, IC 95 % [0.032, 0.089]): allí
  la restricción tapaba también los errores del modelo.
- **El umbral casi no actúa.** Con τ = 0.47 la decisión es la de no restringir salvo en las pocas fotos en que el
  modelo duda entre un mueble del convenio y otro de fuera. La red suele estar segura cuando ve un mueble que el
  convenio no lista.
- **Por qué la restringida da 27 % de falsas alarmas en ODK.** Los otros formularios declaran muebles ganados en
  negociación, que la maestra no lista: la foto muestra lo declarado, pero la decisión restringida no puede decirlo.
  Por eso la corrida la medía solo en las 741 fotos cuyo mueble declarado está en el convenio (0.4 % de falsas
  alarmas y 17 de 17 errores), que es una parte favorable de las 1 148.

**Dos experimentos, no uno.** La detección de 99.3 % y los errores de campo detectados no miden lo mismo:

| | Pares por intercambio | Declaraciones reales de ODK |
|---|---|---|
| Fotos | 3 333 del histórico, de 45 locales con dos o más muebles en el convenio | 1 148 de ODK que declaran un mueble, de 65 locales |
| De dónde sale la discrepancia | Se construye: cada foto se declara como cada otro mueble del convenio | Es real: lo que el mercaderista declaró no es lo que la foto muestra |
| Casos correctos | 3 333 | 1 075 |
| Discrepancias | 5 515, todas construidas | 73, todas errores de campo: 58 de otro mueble, 12 de contenido ajeno y 3 capturas inválidas |
| Detección, con umbral | 5 477 de 5 515 = 99.3 % [0.991, 0.995] | 67 de 73 = 91.8 % [0.832, 0.962] |
| Falsas alarmas, con umbral | 50 de 3 333 = 1.5 % [0.011, 0.020] | 32 de 1 075 = 3.0 % [0.021, 0.042] |

- De las 5 588 discrepancias evaluadas, 73 (1.3 %) son errores reales de campo. En ODK, el 6.4 % de las fotos que
  declaran un mueble no muestra lo declarado.
- En el intercambio la foto siempre es un mueble contratado y bien tomado, declarado como otro: es el caso fácil. El
  99.3 % no es la detección que se espera en el campo.
- La cifra de campo es 67 de 73, con un intervalo ancho porque son 73 errores. De los 6 que no detecta, 5 son otro
  mueble y 1 es contenido ajeno.

Las otras arquitecturas y formatos están en la tabla 5.7 de [`entrenar_tesis.ipynb`](notebooks/entrenar_tesis.ipynb),
con las tres decisiones.

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

El detalle está en `int8_diagnostico.csv` e `int8_capas.csv` de cada run, dentro del paquete de evidencias.

- **El diagnóstico describe, no elige.** Se midió sobre fotos de la prueba porque su única pregunta era por qué falla
  el int8 que ya se había evaluado. Ninguna configuración se eligió ni se ajustó con él, y ninguna de sus variantes
  entra a la tabla 5.7. Si alguna vez se buscara una variante de int8 para usarla, la comparación tendría que hacerse
  con fotos de validación; `tesis.quantize` lo advierte.
- **La hipótesis sobre el int8 completo no se cumple** en MobileNetV3 ni en EfficientNet-B0, y ahí queda. No se sigue
  buscando una configuración int8 que pase los umbrales.
- **float16 es una alternativa, no una prueba de esa hipótesis.** Conserva la exactitud y es el formato que lleva el
  teléfono, pero la hipótesis de la planificación era sobre el int8 completo.

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

  El modelo y el runtime suman 10.0 MB, el 55 % del APK de ODK Collect. Las dos fotos en que el teléfono y la
  evaluación no coinciden se analizan [más abajo](#diferencias-entre-el-teléfono-y-la-evaluación).
- **Desde ODK Collect.** En una copia del formulario de muebles, asignada a un solo usuario, la pregunta de foto abre
  el prototipo y este, Timestamp Camera. Son pruebas de funcionamiento, no mediciones.
  - *15 de noviembre, con el modelo de la corrida anterior.* Una foto oscura de 2560 × 1920 se rechazó como captura
    inválida en 193 ms, volvió al formulario y llegó a Central como adjunto.
  - *17 de noviembre, en un supermercado, con el modelo de la corrida final* (MobileNetV3 en float16; el archivo del
    teléfono es idéntico al evaluado) y el teléfono en modo avión. Fueron dos envíos con un check out declarado, el
    único mueble del convenio de ese local. En el primero, la foto del check out no dio alerta y volvió sola al
    formulario. En el segundo, la foto de un mostrador de frutas dio la alerta «No se ve un mueble» en 210 ms y se
    envió con «Usar de todos modos». Las fotos llegaron a Central al quitar el modo avión. La evaluación en el
    computador decide lo mismo que el teléfono en las dos fotos, a resolución de cámara y en la copia de 1024 px
    de Central. Queda grabada la pantalla de toda la prueba.
  - *Lo que esa prueba dejó ver del convenio.* El prototipo decidía entonces dentro del convenio: con un solo mueble,
    la decisión solo podía ser ese mueble o un rechazo. La foto adicional del segundo envío, un exhibidor de otro
    proveedor que el formulario no pasa por el prototipo, sale en el computador como cabecera (p = 0.81) sin
    restringir y como check out restringida al convenio: con aquella regla no habría dado alerta. Con la decisión
    corregida ([verificación contra el convenio](#verificación-contra-el-convenio)) da alerta de otro mueble. El
    mostrador de frutas también la da, como exhibidor de piso (p = 0.90).
  - *La decisión corregida en el teléfono.* El prototipo se reinstaló el 22 de noviembre de 2026 con la decisión
    nueva (`Clasificador.decidir`, con τ = 0.47) y el mismo modelo; arranca y lo carga. Las mediciones y las dos
    pruebas de arriba son con la decisión restringida. La decisión es una comparación de siete números y no cambia la
    latencia ni la memoria. Una prueba unitaria compila ese método sin Android y comprueba que decide igual que
    `tesis.metrics.decide`; aparte, lo comprobé también sobre las 13 972 fotos de la prueba, cada una con su convenio.
    Con una foto frente a un mueble no se ha probado todavía.
  - *Sin probar en un local:* la alerta de otro mueble, la de una foto movida y repetir la foto tras una alerta.

### Diferencias entre el teléfono y la evaluación

El teléfono y la evaluación dan otra clase en 2 de las 173 fotos medidas, las dos entre las 53 capturas inválidas a
resolución de cámara (3264 px). `python -m tesis.fotos_app --diferencias` repite en Python el preprocesamiento del
teléfono, paso a paso, y compara cada variante con las probabilidades que dio el teléfono:

| Preprocesamiento, sobre las 53 fotos de cámara | Misma clase que el teléfono | Diferencia de probabilidad, mediana | Máxima |
|---|---|---|---|
| Evaluación (`tesis.export` y `tesis.data`) | 51 | 0.024 | 0.133 |
| Evaluación, sin recomprimir la copia de 320 px | 53 | 0.024 | 0.141 |
| Teléfono emulado | 52 | 0.007 | 0.054 |
| Teléfono emulado, enderezando antes de reducir | 52 | 0.007 | 0.054 |
| Teléfono emulado, con el filtro de la evaluación (Lanczos) | 53 | 0.026 | 0.119 |

- **No es la orientación.** Enderezar la foto antes o después de reducirla da las mismas clases y las mismas
  diferencias. Las 173 fotos traen cinco valores de orientación EXIF (0, 1, 3, 6 y 8) y las dos partes los tratan
  igual.
- **Es el redimensionamiento.** La evaluación reduce a 320 px con el filtro Lanczos de PIL; el teléfono, con mitades
  bilineales, que es lo que Android ofrece. Con la reducción del teléfono emulada, la diferencia mediana baja de
  0.024 a 0.007. Si a esa emulación se le cambia solo el filtro por Lanczos, vuelve a 0.026.
- **Y la recompresión.** La evaluación lee la copia de 320 px guardada como JPEG de calidad 92; el teléfono pasa los
  píxeles directo a la red. Sin esa recompresión, la evaluación da la clase del teléfono en las 53 fotos.
- **Lo que queda** (0.007 de mediana) es lo que la emulación no reproduce: el decodificador JPEG y el escalado de
  Android.
- En las 120 fotos como las guarda ODK, todas las variantes dan la clase del teléfono. La diferencia crece con el
  tamaño de la foto porque hay más que reducir.

Las dos fotos son capturas inválidas en que el modelo no se decide: la clase que gana tiene probabilidad 0.35 y
0.34 en el teléfono, y cualquier cambio de píxeles la voltea.

| Foto | Teléfono | Evaluación | Decisión de aceptar o rechazar |
|---|---|---|---|
| 1 | exhibidor de piso (0.348); contenido ajeno, 0.188 | contenido ajeno (0.291); exhibidor de piso, 0.246 | Cambia: el teléfono la acepta como mueble y la evaluación la rechaza |
| 2 | exhibidor de piso (0.343); check out, 0.318 | check out (0.364); exhibidor de piso, 0.315 | No cambia: los dos la aceptan como mueble |

- **La foto 1** la explican los dos pasos: sin recomprimir, la evaluación ya da exhibidor de piso, y la emulación
  del teléfono también.
- **La foto 2** no la reproduce la emulación, que da check out como la evaluación. Las dos clases están a 0.025 en el
  teléfono y la decide lo que la emulación no cubre.
- Ninguna de las dos se reconoce como captura inválida en ninguna de las dos partes. El error de fondo es del modelo,
  no del preprocesamiento.

**Cuándo cambia la decisión.** En las 173 fotos, la decisión de aceptar o rechazar cambia en 1: 0 de las 120 como
las guarda ODK y 1 de las 53 a resolución de cámara (IC 95 % [0.003, 0.099]). Desde ODK la app recibe la foto a
resolución de cámara, así que esa es la condición que importa. De las 165 fotos en que el teléfono da una
probabilidad de 0.5 o más, no difiere ninguna.

**Sobre toda la prueba, con la emulación.** `python -m tesis.fotos_app --emular` pasa las 13 972 fotos de la prueba
por el preprocesamiento del teléfono emulado. No es una medición en el teléfono: es la emulación de arriba, que
reproduce su clase en 172 de las 173 fotos medidas. Frente a la evaluación, con MobileNetV3 en float16:

- La clase cambia en 80 fotos (0.57 %) y la decisión de aceptar o rechazar, en 15 (0.11 %).
- En 13 de esas 15 el preprocesamiento del teléfono acepta lo que la evaluación rechaza, y en 2 es al revés. Nueve
  son fotos de rechazo: 8 de contenido ajeno y 1 captura inválida.
- Las métricas no se mueven: la exactitud es 0.964 con los dos, el recall de captura inválida es el mismo (58 de 90)
  y el de contenido ajeno pasa de 1 279 a 1 276 de 1 326.
- Cambian solo fotos en que el modelo duda. De las 12 787 fotos en que la evaluación da una probabilidad de 0.9 o
  más no cambia ninguna; en las que cambian, la mediana es 0.53.

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
- **Decisión contra el convenio (5.6, 5.9).** La planificación restringe la predicción a los muebles del convenio.
  Así, una foto de un mueble que el convenio no lista pasa por uno contratado. La decisión conserva la clase más
  probable cuando hay evidencia de otro mueble, con un umbral calibrado en validación. La exactitud condicionada de
  la sección 5.6 se sigue reportando, sobre las fotos en que está definida.
- **Deduplicación (5.4).** Se quitan los cuasi-duplicados dentro del local (dhash ≤ 4) en lugar de reconstruir la
  visita. Entre locales se quita toda foto archivada en más de un local (hash idéntico; 672 fotos), porque no se sabe
  de cuál es. Esto se agregó al encontrar 92 fotos de prueba repetidas en otro local, y ahora ocurre antes del sorteo.
- **Partición (5.4).** Los locales se sortearon de nuevo el 16 de noviembre de 2026, después de deduplicar; 18 de los
  112 locales de prueba ya lo eran. Tres locales son de prueba por diseño y no por sorteo: Coral de la Y, Supermaxi
  Iñaquito y Santa María Iñaquito, donde tomé a propósito las capturas inválidas para evaluar. Las capturas inválidas
  reales se estratifican aparte de las sintéticas.
- **Aumentos (5.5).** El cambio de perspectiva se aproxima con zoom y traslación.
- **Máscara (5.5).** Cubre 0.70 × 0.33 ≈ 23 % de la imagen.
- **Tope por local.** `MAX_PER_STORE = 300` no se calibró. Deja fuera del entrenamiento 6 359 de las 47 118 fotos de
  sus locales.
- **Cuantización (5.5).** Además del int8 obligatorio, se prueban float16, int8 de rango dinámico e int8 con
  calibración balanceada.
- **Verificación (5.9).** ODK ya acumula errores reales de campo, así que se adelantó la validación con discrepancias
  reales, que la planificación dejaba como trabajo futuro.
- **Capturas inválidas reales (5.3).** Los mercaderistas están entrenados para no tomarlas, así que casi no hay en el
  histórico (dos). Las 276 de la corrida final vienen de:
  - el formulario *Fotos mal tomadas* de ODK, donde los mercaderistas las tomaron a propósito (las de farmacias quedan
    fuera de alcance);
  - 64 fotos que tomé a propósito el 16 de noviembre de 2026 en locales que por eso son de prueba;
  - las fotos de SOS y Promociones que la revisión manual encontró oscuras, tapadas, movidas o dirigidas al piso.

  Las dos primeras fuentes son capturas inducidas y la tercera, espontáneas. La prueba tiene sobre todo inducidas y
  el entrenamiento, espontáneas; los resultados se reportan [por origen](#capturas-inválidas-inducidas-y-espontáneas).

  Las que son nítidas y muestran otra cosa (una percha, un pasillo, una captura de pantalla) son contenido ajeno,
  como define la sección 2.2, aunque se hayan tomado como ejemplo de foto mala: 98 capturas de pantalla y 11 de mis 64
  fotos quedaron así. En la corrida anterior la prueba tenía 18 capturas inválidas, 16 de ellas sumadas
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
- **Comprobar los resultados sin las fotos ni los modelos.** El paquete reducido de evidencias
  (`python -m tesis.evidencia --reducido`, 19 MB) trae los manifiestos, la partición congelada, los convenios, las
  bitácoras de etiquetado, las predicciones guardadas, las métricas y los registros del teléfono. Descomprimido:

  ```powershell
  pip install numpy pandas scikit-learn
  cd codigo
  python -m tesis.evidencia --tabla ..
  ```

  Recalcula en unos segundos la tabla de resultados: calibra el umbral de cada red con su validación, evalúa cada
  formato sobre la prueba y compara cifra por cifra con `resultados.csv`, la tabla con que se armó el paquete, y con
  las métricas que escribió cada evaluación al terminar de entrenar. Probado en un entorno recién creado, solo con
  esas tres bibliotecas: coinciden las 1 344 cifras.

## Uso de IA

El código se escribió con asistencia de Claude Code (Anthropic). Los prompts que originaron cada parte están como
comentarios `# Prompt (Claude Code): ...` en las celdas de `proyecto_ml.ipynb`. Las decisiones de diseño se justifican
en los cuadernos y en los docstrings de cada módulo.
