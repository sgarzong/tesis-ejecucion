![Logo institucional](planificacion_titulacion_assets/media/6ac6e8f5fcb48728eecb239ba8a51e569530aa54.png)

**COLEGIO DE POSGRADOS**

MAESTRÍA EN CIENCIA DE DATOS

Planificación para el Desarrollo del Proyecto de Titulación

Tutor: Ricardo Flores Moyano

Autor: Sebastián Garzón

Quito – Ecuador

2026

## 1. Título del Proyecto
**Evaluación del compromiso entre desempeño predictivo y eficiencia computacional en la clasificación automática de muebles de exhibición de punto de venta sobre dispositivos móviles de gama baja**

## 2. Relevancia y Justificación
### 2.1 Contexto operativo
Industrias Ales C.A. es una empresa ecuatoriana de consumo masivo fundada el 27 de noviembre de 1943 en Manta, donde inició con la elaboración de jabón de lavar, velas y betún. Con sede actual en Quito y alrededor de novecientos colaboradores, figura entre las ocho empresas de mayor facturación del país según el ranking de las 500 mayores empresas del Ecuador. Su operación se estructura en dos líneas de negocio. La línea industrial produce aceites, mantecas y grasas vegetales, jabones y detergentes, con marcas propias como Ales, Alesol, Girasoli, La Reina y Fab. La línea comercial opera la distribución nacional de marcas representadas de terceros, entre ellas las de Procter & Gamble y Mars.

Esa doble condición de fabricante y distribuidor explica la estructura del problema que aborda este proyecto. El equipo de mercaderistas de Ales ejecuta en el punto de venta tanto los convenios de las marcas propias como los de las marcas representadas, y las maestras de convenio registran esa mezcla en un campo de negocio que distingue entre Ales, Mars, Procter & Gamble e Industrias La Popular (ILP). Un mismo mueble de exhibición puede corresponder a cualquiera de esos negocios y albergar categorías tan distintas como aceites, confitería, cuidado personal, limpieza o alimento para mascotas. El tipo de mueble constituye por tanto la variable de clasificación apropiada, ya que permanece estable con independencia del negocio, de la categoría y de la rotación de marcas y presentaciones que ocupan el mueble.

Sobre esa operación, Industrias Ales mantiene convenios comerciales con cadenas de retail y autoservicios independientes en todo el Ecuador, mediante los cuales se acuerda la instalación de muebles de exhibición específicos en cada local: chicleras, laterales, check outs, cabeceras y exhibidores de piso, entre otros. El cumplimiento de estos convenios se verifica en campo por un equipo de mercaderistas que fotografía cada mueble. Desde septiembre de 2026 ese registro se realiza mediante la herramienta de levantamiento de información ODK Collect; hasta entonces, las fotografías se capturaban con aplicaciones de cámara del teléfono y se cargaban sin clasificar a un repositorio corporativo de OneDrive, origen del corpus histórico descrito en la sección 5.1.

ODK Collect es la aplicación Android del proyecto Open Data Kit, un conjunto de herramientas libres y de código abierto para levantamiento de datos en campo, ampliamente utilizado en investigación, salud pública y trabajo humanitario. Los formularios se diseñan en formato XLSForm, una hoja de cálculo con una fila por pregunta y la lógica de salto y filtrado expresada en columnas, que se convierte al estándar ODK XForms y se publica en un servidor. La aplicación descarga el formulario, opera sin conexión, captura texto, coordenadas, códigos de barras y archivos multimedia, y transmite las sumisiones cuando recupera conectividad. Esa capacidad de operar sin conexión resulta determinante en el levantamiento comercial, ya que buena parte de los locales del país tiene cobertura móvil intermitente.

En la implementación de Industrias Ales, el formulario se apoya en un conjunto de datos externo derivado de la maestra de convenios, que permite filtrar en cascada: la selección del local determina qué opciones de mueble se despliegan a continuación. Cada sumisión, es decir, el registro que el formulario envía al servidor al cerrar una visita, agrupa todas las evidencias levantadas en ella, con una fotografía y un registro de estado por cada mueble contratado.

En la práctica, el mercaderista elige el local que va a levantar y la aplicación despliega la lista de muebles con convenio vigente para ese punto de venta. Sobre cada fila de esa lista adjunta la fotografía correspondiente y registra el estado del mueble, incluyendo el nivel de llenado y la presencia de daños. Cada imagen queda así asociada a un mueble declarado.

El punto vulnerable del proceso se encuentra en esa asociación. Como la lista desplegada restringe las opciones a los muebles contratados para el local, el error posible consiste en adjuntar la fotografía a una fila distinta de la que corresponde, o en capturar una imagen que no permite identificar el mueble. Ese error se propaga hasta los tableros de cumplimiento sin ningún control intermedio, y su detección exige comparar el contenido visual de la imagen con el mueble declarado.

El proyecto propone incorporar una capa de verificación automática que se ejecute en el propio dispositivo del mercaderista, inmediatamente después de la captura y antes del envío al servidor, con dos funciones: contrastar la imagen con el mueble declarado y rechazar las capturas que no constituyen evidencia utilizable. De esta forma, cuando la clasificación predicha discrepa del mueble declarado en el formulario, la persona recibe la alerta mientras todavía se encuentra en la tienda y puede corregir la selección o repetir la captura. Una alerta emitida en el servidor, horas o días después, llega cuando la oportunidad de corrección ya se cerró.

### 2.2 Definición del problema de clasificación
El proyecto se circunscribe al canal moderno de autoservicios, es decir a las cadenas de supermercados e hipermercados. Las farmacias y las estaciones de servicio, que disponen de su propio levantamiento y de un repositorio fotográfico separado, así como el canal tradicional, quedan fuera del alcance y se plantean como extensión posible en el trabajo futuro. Esta delimitación mantiene el corpus, la taxonomía y las maestras de convenio referidos a un mismo universo comercial.

Dentro de ese alcance, el problema se formula como clasificación de imagen completa, de etiqueta única y multiclase, con siete categorías: cinco tipos de mueble de exhibición y dos categorías de rechazo (captura inválida y contenido ajeno), que se describen más adelante en esta sección.

- **Unidad de análisis:** una fotografía capturada por un mercaderista en un punto de venta, que se registra como evidencia de un mueble de exhibición. Cuando la fotografía no pertenece a ninguna de las dos categorías de rechazo, muestra el mueble completo y corresponde a exactamente un mueble.

- **Variable a predecir:** la categoría de la imagen entre siete posibles: uno de los cinco tipos de mueble de exhibición o una de las dos categorías de rechazo.

- **Naturaleza de la etiqueta:** única y mutuamente excluyente. Una misma fotografía puede asociarse a varias marcas o negocios, pero pertenece a un solo tipo de mueble, lo que descarta una formulación multi-etiqueta.

Las clases se obtienen consolidando las nomenclaturas de la maestra de convenios del canal moderno, que hoy registra denominaciones distintas para muebles físicamente equivalentes y desagrega variantes de un mismo mueble. La consolidación fue validada con el área comercial de Industrias Ales y se rige por dos criterios: que las denominaciones agrupadas correspondan al mismo mueble físico, y que las clases resultantes sean distinguibles entre sí en una fotografía, condición que delimita lo que un modelo de visión puede aprender. El resultado son cinco clases que cubren los 572 convenios vigentes en las cadenas de autoservicio. La tabla siguiente presenta la taxonomía; los conteos corresponden a convenios registrados y se incluyen como indicio de la prevalencia relativa esperada.

| **Clase**         | **Nomenclaturas de origen que se consolidan**                | **Convenios** |
|-------------------|--------------------------------------------------------------|---------------|
| Chiclera          | CHICLERA, CHICLERAS                                          | 250           |
| Lateral           | LATERAL SIMPLE, LATERAL DOBLE                                | 164           |
| Check out         | CHECK OUT, CHECK OUT SIMPLE                                  | 80            |
| Cabecera          | CABECERA, CABECERA FRONTAL, CABECERA SECA, CABECERA FASTLINE | 43            |
| Exhibidor de piso | RUMA                                                         | 35            |
| **Total**         |                                                              | **572**       |

A estas cinco clases de mueble se incorporan dos categorías de rechazo, con lo que la tarea queda definida sobre siete categorías en total. Las categorías de rechazo no existen en la nomenclatura comercial, pero son necesarias en operación, ya que una fracción de las capturas de campo no sirve como evidencia. Ambas se construyen durante el etiquetado, a partir del criterio del revisor.

- **Captura inválida:** imagen que no permite identificar ningún mueble por razones técnicas o de encuadre, incluyendo fotografías oscuras o completamente negras, desenfocadas, movidas, o dirigidas al piso o al techo.

- **Contenido ajeno:** imagen legible que muestra algo distinto de un mueble de exhibición, como producto suelto, percha regular sin material de Ales, personas, documentos o exteriores del local.

La detección de estas dos categorías constituye una función del sistema con valor propio, independiente de la verificación contra convenio. Una fotografía negra o dirigida al piso ingresa hoy a los tableros de cumplimiento como evidencia válida, y el mercaderista se entera de la observación cuando ya abandonó la tienda. La alerta en el dispositivo permite repetir la captura en el momento.

El desbalance entre clases es pronunciado: chiclera reúne 250 convenios, mientras que cabecera y exhibidor de piso quedan en 43 y 35 respectivamente, con una razón cercana a siete a uno entre la clase mayoritaria y la minoritaria. El protocolo de etiquetado dirigido descrito en la sección 5.3 prioriza esas dos clases, y el diseño experimental adopta F1-macro como métrica principal porque pondera igual a todas las clases con independencia de su frecuencia.

La verificación operativa se construye sobre esta clasificación y se formula como un problema condicionado al local. Dado que la sumisión de ODK aporta tanto el local como el mueble declarado, la decisión consiste en determinar si la imagen corresponde a esa declaración, dentro del conjunto acotado de muebles contratados para ese punto de venta. La condición reduce el espacio de decisión: en lugar de discriminar entre las cinco clases de mueble, el sistema distingue entre los dos o tres muebles que el local tiene bajo convenio, además de las dos categorías de rechazo.

De los 338 locales de autoservicio presentes en la maestra, 203 tienen dos o tres convenios y son aquellos en los que una selección equivocada dentro de la lista resulta posible. En los 135 locales de convenio único, es decir, con un solo tipo de mueble contratado, la lista ofrece una sola opción, de modo que el error de selección queda descartado por construcción. El control de calidad de la captura opera en cambio sobre la totalidad de los locales, ya que una fotografía inválida o de contenido ajeno puede producirse en cualquier visita, con independencia del número de convenios.

### 2.3 Contribución esperada
La contribución del proyecto es un estudio experimental que determina qué configuración de arquitectura y técnica de compresión ofrece el mejor compromiso entre desempeño predictivo y costo computacional bajo restricciones de hardware reales y verificables: un dispositivo concreto en uso en campo, un presupuesto de latencia derivado de la operación y un umbral de desempeño derivado del costo del error. El etiquetado débil derivado de la maestra de convenios se emplea como estrategia metodológica para construir el conjunto de datos y no constituye una contribución evaluada del proyecto; su tasa de error estimada se reporta como característica del conjunto (sección 5.3). La aplicación móvil cumple aquí la función de instrumento de medición y constituye un medio para responder las preguntas de investigación.

## 3. Objetivos
### 3.1 Preguntas de investigación
El proyecto se organiza alrededor de tres preguntas que guían los experimentos y las conclusiones:

1.  ¿Qué diferencia de desempeño predictivo existe entre una arquitectura profunda convencional, empleada como línea base, y arquitecturas diseñadas para eficiencia, en la clasificación del tipo de mueble de exhibición sobre fotografías capturadas en condiciones reales de punto de venta?

2.  ¿Cuánto desempeño predictivo se pierde y cuánta latencia, memoria y tamaño de modelo se ganan al aplicar cuantización post-entrenamiento a int8 para inferencia local en el dispositivo de referencia?

3.  ¿Qué configuraciones de arquitectura y compresión satisfacen simultáneamente los umbrales operativos definidos, y cuál de ellas ofrece el mejor compromiso entre desempeño y costo computacional?

### 3.2 Objetivo general
Evaluar experimentalmente el compromiso entre desempeño predictivo y eficiencia computacional de arquitecturas de aprendizaje profundo aplicadas a la clasificación automática del tipo de mueble de exhibición en fotografías de punto de venta, determinando qué configuración de modelo y técnica de compresión satisface los umbrales operativos de desempeño predictivo, latencia, memoria y tamaño establecidos para inferencia local en un dispositivo móvil de gama baja en uso en campo.

### 3.3 Objetivos específicos
1.  Construir y documentar un conjunto de datos etiquetado de fotografías de muebles de exhibición, definiendo la taxonomía de clases, el protocolo de etiquetado, la estrategia de etiquetado débil derivada de la maestra de convenios y los controles de calidad aplicados.

2.  Diseñar una partición de datos agrupada por local que impida la fuga de información entre entrenamiento y prueba, incorporando deduplicación perceptual, y evaluar por separado la generalización a locales no vistos y la generalización temporal mediante dos experimentos distintos.

3.  Entrenar mediante aprendizaje por transferencia un modelo de línea base y dos arquitecturas eficientes bajo un protocolo experimental común, y comparar su desempeño predictivo.

4.  Cuantificar el efecto de la cuantización post-entrenamiento sobre el desempeño predictivo y sobre las métricas de eficiencia, medidas directamente en el dispositivo de referencia.

5.  Implementar un prototipo de inferencia local que clasifique la fotografía capturada, la contraste con el mueble declarado en el formulario y advierta sobre capturas inválidas o de contenido ajeno, sin depender de conexión a internet.

6.  Analizar los errores del sistema por clase y por cadena, identificando clases confundibles, condiciones de captura problemáticas y limitaciones para el escalamiento.

## 4. Estado del Arte
El reconocimiento visual aplicado a retail se ha concentrado en el reconocimiento de productos individuales. Tonioni, Serra y Di Stefano \[14\] proponen una tubería de reconocimiento de productos en percha basada en detección seguida de emparejamiento por similitud, evaluada sobre escenas de góndola. Wei y colaboradores \[15\] publican RPC, un conjunto de datos a gran escala para checkout automático con anotación a nivel de producto. Goldman y colaboradores \[16\] abordan la detección en escenas densamente pobladas, donde el reto es separar instancias muy próximas entre sí. Peng, Xiao y Li \[17\] presentan RP2K, orientado a clasificación de grano fino sobre miles de referencias comerciales. Wei y colaboradores \[18\] sistematizan en una revisión los desafíos recurrentes del dominio: similitud visual entre empaques, cambios frecuentes de diseño, oclusiones, reflejos y variabilidad de captura.

En paralelo, la línea de investigación sobre modelos eficientes ha producido arquitecturas diseñadas explícitamente para el compromiso entre exactitud y costo. Sandler y colaboradores \[3\] introducen los bloques residuales invertidos de MobileNetV2; Howard y colaboradores \[4\] refinan el diseño en MobileNetV3 mediante búsqueda de arquitectura; Tan y Le formalizan el escalamiento compuesto en EfficientNet \[5\] y posteriormente optimizan la velocidad de entrenamiento en EfficientNetV2 \[6\]. Trabajos más recientes desplazan el criterio de diseño desde el conteo de operaciones hacia la latencia medida en dispositivo: Vasu y colaboradores \[7\] optimizan MobileOne contra el tiempo de inferencia real en hardware móvil, Mehta y Rastegari \[8\] combinan convoluciones y atención en MobileViT, y Qin y colaboradores \[9\] presentan MobileNetV4 como familia orientada al ecosistema móvil actual. Sobre compresión, Jacob y colaboradores \[10\] establecen el esquema de cuantización a aritmética entera con simulación durante el entrenamiento, Gholami y colaboradores \[11\] sistematizan el campo en una revisión, y Hinton, Vinyals y Dean \[12\] formulan la destilación de conocimiento.

La limitación que este proyecto aborda se ubica en la intersección de ambas líneas. Los trabajos de retail toman como unidad de análisis el producto o la referencia comercial y evalúan sobre infraestructura de servidor, sin reportar restricciones de despliegue; el mueble de exhibición como objeto a clasificar, y la validación de la evidencia contra una declaración administrativa previa, no aparecen tratados en esa literatura. Los trabajos de eficiencia, a su vez, reportan latencia sobre dispositivos de gama alta o media y sobre conjuntos de datos de referencia como ImageNet, no sobre un dominio industrial con desbalance severo, etiquetas ruidosas y un dispositivo de gama baja fijado por la operación. El aporte consiste en producir esa evidencia empírica faltante bajo restricciones reales y verificables. Para construir el conjunto de datos se recurre además al etiquetado débil derivado de registros administrativos, apoyado en la literatura sobre aprendizaje con ruido de etiqueta sistematizada por Northcutt, Jiang y Chuang \[19\]; se trata de una decisión metodológica de construcción del dataset y no de un objeto de evaluación del proyecto.

## 5. Metodología de Trabajo
### 5.1 Caracterización del conjunto de datos
El proyecto emplea dos corpus de naturaleza distinta, originados en dos procesos de captura sucesivos.

Corpus histórico. Reúne las fotografías anteriores a la puesta en marcha de ODK Collect, levantadas de forma individual y cargadas por los propios mercaderistas al repositorio corporativo de OneDrive. Las limitaciones de ese esquema de carga llevaron a que las imágenes se subieran sin clasificar, de modo que el mueble representado no quedó registrado en ningún campo. Este corpus constituye la fuente de entrenamiento del proyecto y fue inventariado de forma exhaustiva durante la fase de planificación. Sus características son las siguientes.

- **Volumen:** 64 415 fotografías.

- **Periodo:** enero de 2024 a diciembre de 2025, con 23 meses poblados. Mayo de 2024 no registra fotografías. El volumen mensual oscila entre 2 084 (febrero de 2025) y 3 615 (julio de 2024), con una media aproximada de 2 800. Las fotografías de enero a agosto de 2026, anteriores a la puesta en marcha de ODK, no están disponibles para el proyecto.

- **Procedencia:** levantamiento realizado por mercaderistas de Industrias Ales en el marco de la verificación de convenios comerciales, antes de la adopción de ODK Collect. Las fotografías se capturaron con la cámara del teléfono, mayoritariamente mediante Timestamp Camera Free, y se cargaron manualmente al repositorio de OneDrive.

- **Cobertura:** cinco cadenas de autoservicio y 931 puntos de venta (PDV) distintos, incluidos 48 locales cuya cadena no pudo asignarse, distribuidos según la tabla siguiente.

| **Cadena**                          | **Fotografías** | **%**     | **PDV** | **Media de fotos por PDV** |
|-------------------------------------|-----------------|-----------|---------|----------------------------|
| Tiendas Industriales Asociadas TIA  | 30 295          | 47,0      | 397     | 76                         |
| Coral Hipermercados (Gerardo Ortiz) | 12 887          | 20,0      | 62      | 208                        |
| Mega Santamaría                     | 8 392           | 13,0      | 83      | 101                        |
| Corporación El Rosado               | 8 339           | 12,9      | 159     | 52                         |
| Corporación Favorita                | 4 058           | 6,3       | 182     | 22                         |
| Sin clasificar                      | 444             | 0,7       | 48      | 9                          |
| **Total**                           | **64 415**      | **100,0** | **931** | **69**                     |

- **Distribución por punto de venta:** fuertemente desigual entre cadenas. Con 931 puntos de venta, la media global asciende a 69 fotografías por local, pero oscila entre 22 en Favorita y 208 en Coral, según la tabla anterior. La cola alta corresponde íntegramente a Coral: agregando por nombre de local a lo largo de los 23 meses, Manta reúne 1 158 imágenes, el máximo del corpus, seguida de Monay Shopping con 1 091, Carapungo con 974, Ceibos con 872 y Guayaquil con 770.

- **Estructura de almacenamiento:** la jerarquía nominal es cadena, supervisor, mercaderista, punto de venta y archivo, con 187 supervisores y 1 177 mercaderistas. La profundidad es irregular: 58 593 archivos siguen esa estructura exacta, 4 101 incorporan un nivel adicional de semana, 685 añaden semana y material, 153 alcanzan ocho niveles y 96 se ubican entre el primer y el tercer nivel.

- **Calidad de la asignación de cadena:** 61 444 fotografías, equivalentes al 95,4 %, toman la cadena directamente de la carpeta raíz. Las 2 527 restantes la infieren del nombre del local, y de ellas 1 859 se asignaron mediante un código numérico que emplean tanto TIA como Favorita, lo que convierte a ese 2,9 % en el segmento de menor confianza. Las 444 fotografías sin clasificar corresponden a nombres de localidad sin marca que, por geografía y por convivencia con carpetas de Santamaría, se presumen mayoritariamente de esa cadena.

- **Evolución del mix de cobertura:** TIA crece 40 % entre 2024 y 2025, mientras que las otras cuatro cadenas retroceden entre 20 % y 30 %. El volumen total se mantiene estable, de modo que se trata de un cambio en la composición de la cobertura.

- **Dispositivos de captura:** flota corporativa de teléfonos Android, renovada parcialmente en 2025. Las fotografías anteriores a esa renovación provienen de equipos más antiguos y de menor calidad óptica, lo que introduce una fuente adicional de variabilidad.

- **Aplicación de captura:** la mayor parte de las fotografías fue tomada con Timestamp Camera Free, una aplicación que superpone sobre la imagen, en la esquina inferior derecha, la fecha, la hora y las coordenadas geográficas del momento de captura. Esa superposición queda incorporada a los píxeles y sobrevive a la recodificación que aplica el repositorio, de modo que constituye la única fuente de metadatos disponible por fotografía.

- **Etiquetas:** no existen. Las fotografías del corpus histórico se cargaron sin clasificar, de modo que la totalidad del etiquetado debe construirse en el marco del proyecto.

- **Restricciones de confidencialidad:** no existen limitaciones para el uso académico de las imágenes.

Corpus de ODK. Reúne las sumisiones generadas desde la puesta en marcha del formulario, en septiembre de 2026, y se almacena en un repositorio en la nube (un bucket de Amazon S3) con un nombre de archivo que permite unir cada imagen con su sumisión de origen. Cada fotografía conserva el mueble declarado, el local, el usuario, la fecha, el nivel de llenado y el estado de daño, información ausente en el corpus histórico. Al igual que en el corpus histórico, las fotografías se capturan con Timestamp Camera Free, aplicación de cámara externa que Industrias Ales exige a sus mercaderistas, de modo que todas conservan la superposición de fecha, hora y coordenadas.

El levantamiento opera por ciclos mensuales y cubre la totalidad del universo de locales en cada ciclo. Restringido al canal moderno, ese universo comprende 338 locales y 572 convenios, de modo que un ciclo completo produce del orden de 570 fotografías con mueble declarado. A la fecha de esta planificación se han levantado 110 locales del primer ciclo, repartidos entre todos los canales. Considerando el periodo de puesta en marcha, el proyecto dispondrá al cierre de un volumen aproximado de dos mil fotografías con declaración asociada.

Ese tamaño resulta insuficiente para entrenamiento y adecuado para tres funciones en el diseño: conformar el conjunto de evaluación en condiciones de operación real, complementar en el entrenamiento las clases minoritarias y las categorías de rechazo, respetando la asignación de locales de la sección 5.4, y alimentar el experimento de verificación de la sección 5.9. Las etiquetas se verifican manualmente antes de su uso, dado que la declaración del mercaderista constituye precisamente el objeto de la verificación. La recurrencia mensual implica que este corpus crece de forma sostenida y que, superado el horizonte del proyecto, desplazará al histórico como fuente principal de entrenamiento.

### 5.2 Cobertura del corpus y riesgos identificados
El inventario del repositorio revela siete condiciones que afectan el diseño y que se documentan de forma explícita.

1.  **Cobertura parcial de la maestra de convenios sobre el corpus.** Corporación Favorita aporta 4 058 fotografías, el 6,3 % del corpus, y no figura en las maestras vigentes, porque el convenio con esa cadena ya no está activo; sus fotografías corresponden al periodo en que sí lo estaba. Esas imágenes se conservan para la tarea de clasificación, con etiquetado exclusivamente manual, y se excluyen tanto de la propagación de etiquetas débiles como del experimento de verificación descrito en la sección 5.9. Su aporte de diversidad es alto, ya que corresponden a 182 puntos de venta distintos. Las 444 fotografías sin clasificar reciben el mismo tratamiento hasta que se confirme su cadena.

2.  **Alcance restringido de la taxonomía.** Las clases ristra, OTA (mueble que se cuelga del techo para organizar productos en sachet) y exhibidor de counter corresponden a farmacias y al canal tradicional, ambos fuera del alcance definido en la sección 2.2, y por tanto no forman parte de la taxonomía. La verificación de que el corpus del canal moderno contiene efectivamente ejemplares de las cinco clases retenidas, y en particular de exhibidor de piso, es la primera actividad del proyecto.

3.  **Desfase temporal entre convenios y fotografías.** Las maestras disponibles corresponden a la vigencia 2026, mientras que las fotografías abarcan 2024 y 2025. La propagación de etiquetas débiles descrita en la sección 5.3 descansa en el supuesto de que el mueble contratado por local se mantuvo estable entre ambos periodos. Ese supuesto debe validarse recuperando las maestras de los años correspondientes; en su defecto, la estimación del ruido de etiqueta debe incorporar esta fuente de error de forma explícita.

4.  **Ausencia de una clave de unión directa.** Las carpetas identifican el local por nombre, mientras que la maestra lo identifica por código (formato UTT seguido de un número) y por una denominación distinta. La reconciliación se apoyará en dos vías complementarias: la extracción por reconocimiento óptico de caracteres (OCR) de las coordenadas superpuestas en cada fotografía, contrastadas contra las coordenadas registradas en la maestra, y una tabla de correspondencia entre nomenclaturas con verificación manual de los casos ambiguos. La maestra dispone de coordenadas utilizables para 301 de los 338 locales, el 89 %, y la separación entre locales resulta suficiente para identificarlos de forma unívoca: la distancia mediana al local más próximo asciende a 1 071 metros y solo el 10 % tiene un vecino a menos de 30 metros, casos que corresponden a locales dentro de un mismo centro comercial y que se resolverán por nombre. Conviene notar que el corpus abarca más puntos de venta (931) que la maestra vigente (338). La diferencia obedece, al menos en parte, a los 182 locales de Favorita, cuyo convenio ya no está vigente, a los 48 sin cadena asignada y al desfase temporal descrito en el punto anterior, ya que una maestra de 2026 puede no incluir locales cuyo convenio dejó de estar vigente. El número de locales del corpus que se emparejan con la maestra, y por tanto de imágenes que reciben etiqueta débil, se determinará durante la reconciliación (actividad A3).

5.  **Concentración extrema por local.** Cinco tiendas de Coral concentran cerca de 4 900 fotografías, el 7,6 % del corpus. Sin control, esos locales dominarían el entrenamiento y sus condiciones particulares de iluminación y disposición se transferirían al modelo. Se acotará el número máximo de imágenes por local en el conjunto de entrenamiento.

6.  **Irregularidad de la jerarquía de carpetas.** Tomar la última carpeta antes del archivo como identificador de local produce 3 761 identificadores para 931 tiendas reales, porque las rutas con niveles adicionales de semana o material desplazan el nivel correcto. La normalización se realizará por nombre de local y no por ruta, condición indispensable para que el agrupamiento de la sección 5.4 cumpla su función.

7.  **Cambio en la composición de la cobertura entre años.** El crecimiento de TIA y el retroceso de las demás cadenas implica que el conjunto de prueba del experimento temporal (experimento B, sección 5.4) difiere del de entrenamiento en su mezcla de cadenas. Los resultados de ese experimento se interpretarán considerando ese desplazamiento.

### 5.3 Estrategia de etiquetado
El cronograma disponible admite un volumen de etiquetado manual muy inferior a las 64 000 imágenes del corpus. La estrategia combina tres mecanismos y se documenta como decisión metodológica explícita:

1.  **Propagación de etiquetas débiles.** Para los 135 locales de autoservicio de convenio único, la etiqueta se propaga automáticamente desde la maestra de convenios, sin intervención manual, una vez que cada local del corpus se ha reconciliado con su código en la maestra por los procedimientos descritos en la sección 5.2 (punto 4). La etiqueta propagada no distingue las capturas inválidas o de contenido ajeno que también se producen en esos locales; la muestra de verificación del punto siguiente cuantifica también esta fuente de error. Ese bloque cubre cuatro de las cinco clases, con 60 locales de chiclera, 44 de check out, 30 de cabecera y uno de lateral. La clase exhibidor de piso carece de locales de convenio único, de modo que su totalidad depende del etiquetado manual.

2.  **Estimación del ruido de etiqueta.** Se verifica manualmente una muestra aleatoria de ese bloque para estimar la tasa de error de la etiqueta débil, valor que se reporta como característica del conjunto de datos y se emplea para acotar la interpretación de los resultados.

3.  **Etiquetado manual dirigido.** El esfuerzo humano se concentra en los 203 locales con dos o tres convenios, en los locales de Favorita sin convenio asociado, y en las clases exhibidor de piso y lateral, que el bloque de etiqueta débil cubre de forma insuficiente. Dado que la propagación automática cubre solo los 135 locales de convenio único, el etiquetado manual constituye la vía principal de construcción del conjunto. Con una dedicación de 24 horas semanales durante doce semanas, de septiembre a noviembre de 2026 según el cronograma de la sección 8 (cerca de 290 horas), y un ritmo estimado de 20 a 25 imágenes por hora incluida la revisión, se proyecta etiquetar o verificar manualmente del orden de 6 000 imágenes. Ese volumen incluye la muestra de control del bloque de etiqueta débil y resulta suficiente para conformar particiones de validación y prueba con presencia adecuada de las siete categorías.

El conjunto de prueba se construirá exclusivamente con imágenes de verificación manual, de modo que las métricas reportadas dependan únicamente de etiquetas confirmadas por revisión humana.

Las dos categorías de rechazo requieren un tratamiento aparte. Su prevalencia en el corpus histórico es desconocida y se estimará durante la primera iteración de etiquetado, con el propósito de dimensionar el esfuerzo necesario para reunir ejemplares suficientes. La revisión manual las marcará de forma sistemática cada vez que aparezcan, sin descartarlas como ruido. Las condiciones de oscuridad, desenfoque y movimiento admiten síntesis controlada a partir de imágenes válidas, lo que permite ampliar esa porción del conjunto; los errores de encuadre, como las fotografías dirigidas al piso o al techo, y el contenido ajeno solo pueden obtenerse de ejemplares reales, de modo que su recolección condiciona el calendario de etiquetado. Si el corpus histórico no aporta un número suficiente, se recurrirá a las sumisiones de ODK acumuladas durante el proyecto, cuyo ritmo mensual las convierte en una fuente creciente para estas categorías.

### 5.4 Partición de datos y control de fuga
Este dominio exige una estrategia de partición más estricta que la estratificación simple. Un mismo local es visitado repetidamente a lo largo de dos años, de modo que fotografías del mismo mueble, con el mismo fondo y bajo condiciones de iluminación similares, aparecerían simultáneamente en entrenamiento y prueba, inflando artificialmente el desempeño reportado, un riesgo documentado de forma sistemática por Kapoor y Narayanan \[20\]. Se aplicarán los controles siguientes, y la evaluación se organiza en dos experimentos distintos, cada uno con su propia partición:

- Experimento A, generalización a locales no vistos (experimento principal). Partición agrupada por local sobre la totalidad del periodo, que incluye tanto el corpus histórico como las fotografías de ODK. Estas últimas se incorporan al experimento porque desde septiembre de 2026 rigen nuevos convenios comerciales, de modo que representan la configuración vigente de muebles en los locales. La partición emplea el nombre normalizado del punto de venta como clave de agrupamiento. Ningún local aparece en más de una partición. El conjunto de prueba se compone de locales nunca vistos durante el entrenamiento, replicando la condición que enfrenta el sistema en producción. Los criterios predictivos de la sección 5.7 y las hipótesis de la sección 5.8 se evalúan sobre este experimento, con excepción de la tasa de falsas alarmas, que se mide en la evaluación de verificación de la sección 5.9.

- Experimento B, generalización temporal (experimento de robustez). Se entrena con las fotografías de enero de 2024 a septiembre de 2025 y se evalúa sobre el último trimestre del corpus histórico, de octubre a diciembre de 2025. La asignación de locales del experimento A se mantiene: el entrenamiento emplea solo locales de entrenamiento y la prueba solo locales de prueba, de modo que la comparación entre ambos experimentos aproxima el efecto del desplazamiento temporal (cambios de material de exhibición, de flota de dispositivos y de mezcla de cadenas). La aproximación no aísla ese efecto por completo, porque el experimento B entrena con menos meses de datos y evalúa sobre un conjunto más pequeño; esta limitación se considera al interpretar los resultados. Los modelos se reentrenan para este experimento con el mismo protocolo, y sus resultados se reportan por separado.

- Deduplicación perceptual y agrupamiento por visita. Se calculan huellas perceptuales sobre todas las imágenes para identificar duplicados exactos y cuasi-duplicados. La marca de tiempo recuperada por reconocimiento óptico permite además reconstruir la visita, entendida como el conjunto de capturas realizadas en un mismo local dentro de una ventana breve, que constituye la unidad natural de agrupamiento para las ráfagas de fotografías de una misma escena.

- Consistencia entre corpus. La asignación de cada local a una partición se mantiene al incorporar las fotografías de ODK, y los locales que aparecen por primera vez en ODK se asignan con el mismo criterio de agrupamiento. Como el levantamiento recorre el mismo universo de locales cada mes, un local asignado a prueba en el corpus histórico conserva esa condición en el corpus de ODK, lo que impide que la recurrencia mensual reintroduzca la fuga que el agrupamiento evita.

La estratificación por clase se aplica dentro de la restricción de agrupamiento, priorizando que las clases minoritarias mantengan presencia suficiente en validación y prueba.

### 5.5 Alcance experimental
El alcance se restringe deliberadamente para permitir un análisis profundo de un número acotado de configuraciones. La comparación principal contempla tres modelos, todos inicializados con pesos preentrenados en ImageNet \[2\] y entrenados sobre la misma partición y bajo un protocolo idéntico de aumentos de datos, optimizador, tasa de aprendizaje y criterio de parada temprana:

- ResNet-50 como línea base de referencia, sin restricción de eficiencia.

- MobileNetV3-Large como arquitectura eficiente representativa del diseño orientado a móviles.

- EfficientNet-B0 como segunda arquitectura eficiente, con un compromiso distinto entre profundidad y anchura.

La única técnica de compresión de aplicación obligatoria es la cuantización post-entrenamiento a int8, aplicada a los tres modelos con un conjunto representativo de calibración, siguiendo el procedimiento estándar de TensorFlow Lite \[13\]. La cuantización consciente del entrenamiento y la destilación de conocimiento se declaran explícitamente como líneas opcionales, que se abordarán solo si el cronograma lo permite y si la cuantización post-entrenamiento no alcanza los umbrales establecidos. Las tres preguntas de investigación se responden en su totalidad con la cuantización post-entrenamiento.

Los aumentos de datos se limitan a transformaciones que reproducen condiciones reales de captura en tienda: variación de brillo y contraste, rotación leve, recorte aleatorio, cambio de perspectiva y compresión con pérdida. La superposición de fecha, hora y coordenadas se enmascara en el recorte de entrada de forma sistemática. El texto incorporado contiene las coordenadas del local, que identifican el punto de venta con precisión y actúan como un identificador legible dentro de la imagen; un modelo convolucional puede aprender a discriminar por esa señal y alcanzar un desempeño aparente elevado sin observar el mueble. El enmascaramiento de la región inferior derecha se aplica por tanto en todas las particiones y en ambos corpus, ya que las fotografías de ODK, y las que el sistema recibirá en operación, también se capturan con Timestamp Camera Free y llevan la misma superposición, y se documenta la pérdida de superficie útil que implica. Como control, se entrenará una variante con la superposición visible para cuantificar la magnitud del atajo.

### 5.6 Métricas
Las métricas se reportan en dos bloques separados, y cada bloque conserva sus propias unidades e interpretación.

**Métricas predictivas, calculadas sobre el conjunto de prueba del experimento A (locales no vistos) y reportadas por separado para el experimento B (evaluación temporal):**

- F1-macro como métrica principal, apropiada ante desbalance severo porque pondera igual a todas las clases.

- Precision y recall por clase, reportados de forma individual para cada una de las siete categorías, con atención particular al recall de las dos categorías de rechazo.

- Matriz de confusión normalizada, para identificar los pares de clases sistemáticamente confundidos.

- Recall-macro como métrica complementaria del F1-macro, con umbral de viabilidad propio en la sección 5.7.

- Exactitud condicionada al local, calculada restringiendo la predicción al conjunto de muebles contratados para el punto de venta más las dos categorías de rechazo, que corresponde a la decisión efectiva del sistema en operación.

- Tasa de detección de discrepancias y tasa de falsas alarmas sobre el experimento de verificación descrito en la sección 5.9.

**Métricas de eficiencia, medidas empíricamente sobre el dispositivo de referencia:**

- Latencia de extremo a extremo, desde que la imagen capturada está disponible hasta la emisión de la predicción, incluido el preprocesamiento, reportada como mediana y percentil 95 sobre un mínimo de cien ejecuciones, organizadas en ráfagas continuas de 30 inferencias según el protocolo de carga sostenida descrito en la sección 5.7.

- Memoria pico en tiempo de inferencia, en megabytes.

- Tamaño del modelo desplegado, en megabytes, y su incremento sobre el tamaño del paquete de la aplicación.

### 5.7 Criterios cuantitativos de éxito
El dispositivo de referencia es el Honor X6b, equipo de la flota corporativa incorporado en la renovación de 2025, con procesador MediaTek Helio G85 de 12 nm (ocho núcleos: dos Cortex-A75 a 2.0 GHz y seis Cortex-A55 a 1.8 GHz), GPU Mali-G52 MC2, 6 GB de memoria RAM y Android 14 sobre MagicOS 8. Todas las mediciones de eficiencia se realizan sobre ese equipo y bajo condiciones equivalentes a las de campo, con la aplicación de levantamiento activa.

Las características de este equipo condicionan el diseño experimental de tres maneras, que conviene explicitar porque determinan qué resultados son alcanzables.

Dónde se ejecuta la inferencia. Los teléfonos de gama media y alta incorporan una unidad de procesamiento neuronal, un bloque de silicio especializado en las operaciones de convolución y multiplicación de matrices, al que las aplicaciones acceden mediante la interfaz NNAPI de Android. El Helio G85 carece de ese bloque. Su GPU Mali-G52 MC2 dispone de dos núcleos de ejecución y está dimensionada para carga gráfica, con soporte parcial e inestable para tareas de inferencia. La ejecución recae por tanto sobre los ocho núcleos ARM del procesador, a través de XNNPACK, la biblioteca de núcleos de cómputo optimizados que TensorFlow Lite emplea para CPU. Toda la latencia reportada corresponde a ese camino de ejecución.

Qué aporta la cuantización en este contexto. Convertir los pesos de coma flotante de 32 bits a enteros de 8 bits reduce aproximadamente a la cuarta parte el espacio que ocupan los pesos y, con ello, el tamaño del archivo del modelo. Esa proporción no se traslada de forma directa a la memoria total de inferencia, que incluye además los tensores de activación intermedios, los búferes del intérprete y de XNNPACK y la carga fija del runtime; aun con cuantización completa, algunas operaciones y las capas de entrada y salida pueden permanecer en coma flotante. Por esa razón, la reducción efectiva de la memoria pico no se asume a priori y se mide sobre el dispositivo. La ganancia de velocidad proviene de las instrucciones vectoriales del conjunto ARM NEON, cuyos registros de 128 bits alojan cuatro veces más valores enteros de 8 bits que valores en coma flotante de 32 bits. La magnitud efectiva de esa ganancia depende de la arquitectura concreta del procesador y de la proporción de operaciones que el compilador logra vectorizar, lo que obliga a medirla de forma empírica sobre el equipo y explica por qué la segunda pregunta de investigación se formula en términos de latencia observada. Se reportará, como control, la latencia con y sin cuantización para dimensionar la ganancia atribuible a esta técnica.

Comportamiento térmico y protocolo de medición. El Helio G85 se fabrica en un nodo de 12 nanómetros, con una disipación de calor mayor que la de los procesadores de 6 o 4 nanómetros de generaciones recientes. Bajo carga sostenida, el sistema operativo reduce la frecuencia de reloj de los núcleos para contener la temperatura, mecanismo conocido como estrangulamiento térmico. Dado que un mercaderista captura entre 20 y 30 fotografías a lo largo de su jornada, la medición se realizará en una ráfaga continua de 30 inferencias, escenario más exigente que el uso real porque concentra en pocos minutos la carga de un día completo, y se reportarán de forma separada la latencia de la primera inferencia y la de la trigésima, de modo que la degradación por temperatura quede documentada. Los umbrales definidos más adelante deben cumplirse en esa condición sostenida.

El costo del error es asimétrico, y la omisión de una discrepancia resulta más costosa que una falsa alarma: una fotografía mal catalogada que llega al tablero compromete la confianza de la gerencia en la información, mientras que una alerta injustificada solo obliga a repetir una captura. Al mismo tiempo, con un volumen de 20 a 30 fotografías diarias por mercaderista, una tasa elevada de falsas alarmas volvería la herramienta inutilizable en la práctica. En la verificación, esto se traduce en privilegiar la tasa de detección de discrepancias sujeta a una restricción explícita sobre las falsas alarmas. En la clasificación, se traduce en exigir un recall alto en todas las clases, de modo que ninguna, y en particular las minoritarias, quede con baja capacidad de detección.

| **Criterio**                        | **Umbral**                      | **Justificación**                                                                                    |
|-------------------------------------|---------------------------------|------------------------------------------------------------------------------------------------------|
| F1-macro (versión desplegada)       | ≥ 0.85 en el conjunto de prueba | Desempeño mínimo para que la alerta sea confiable en las clases minoritarias                         |
| Recall-macro (versión desplegada)   | ≥ 0.90                          | Ninguna clase, incluidas las minoritarias, puede quedar con baja capacidad de detección              |
| Tasa de falsas alarmas              | ≤ 5 % de los pares correctos    | Con 20 a 30 fotos diarias equivale a una o dos interrupciones por día                                |
| Recall de cada categoría de rechazo | ≥ 0.90                          | Una evidencia inservible que llega al tablero degrada la información de cumplimiento                 |
| Degradación por cuantización        | ≤ 2 puntos de F1-macro          | Límite de pérdida aceptable frente a la ganancia en latencia y tamaño                                |
| Latencia mediana en dispositivo     | ≤ 2 000 ms de extremo a extremo | Tiempo tolerable entre la captura y la alerta sin entorpecer la visita                               |
| Latencia p95 en dispositivo         | ≤ 3 000 ms                      | Acota la cola de la distribución, además del caso típico                                             |
| Memoria pico en inferencia          | ≤ 300 MB                        | Presupuesto compartido con la aplicación de levantamiento y la cámara sobre los 6 GB del dispositivo |
| Tamaño del modelo desplegado        | ≤ 20 MB                         | La actualización se descarga con datos móviles en campo                                              |

Los umbrales predictivos se evalúan sobre la versión que se despliega en el dispositivo, es decir, la cuantizada a int8 cuando corresponda; la degradación por cuantización se calcula como la diferencia de F1-macro entre las versiones float32 e int8 del mismo modelo. Las diferencias de F1-macro se expresan en puntos, donde un punto equivale a 0,01. La tasa de falsas alarmas se define como la proporción de pares correctos de la evaluación de verificación (sección 5.9), es decir, fotografías válidas cuya declaración coincide con el mueble, para los que el sistema emite cualquier alerta, sea de discrepancia o de rechazo, aplicando la decisión condicionada al local. El umbral se evalúa sobre los pares construidos con los locales de prueba del experimento A; la evaluación sobre las sumisiones de ODK se reporta como complemento. Una configuración se considera operativamente viable cuando satisface la totalidad de los criterios de la tabla de forma simultánea.

Entre las configuraciones viables, la que representa el mejor compromiso entre desempeño y eficiencia se elige mediante un criterio lexicográfico fijado de antemano. Primero se prefiere la de mayor F1-macro de la versión desplegada, en int8 cuando corresponda, sobre el conjunto de prueba del experimento A. Toda configuración cuyo F1-macro quede a menos de un punto (0,01) de la mejor, o cuya diferencia con la mejor tenga un intervalo de confianza bootstrap al 95 % (con remuestreo por local) que incluya el cero, criterio coherente con el de la sección 5.8, se considera equivalente en desempeño y se prefiere la de menor latencia p95 en el dispositivo. Si el empate persiste, decide la menor memoria pico. Como complemento, se reporta el frente de Pareto de F1-macro frente a latencia para todas las configuraciones evaluadas, viables o no. Si ninguna configuración resulta viable, no se declara una configuración recomendada y se reporta cuál de los criterios impide la viabilidad en cada caso.

### 5.8 Hipótesis de trabajo
*Una arquitectura eficiente entrenada mediante aprendizaje por transferencia y cuantizada a int8 satisfará la totalidad de los criterios de viabilidad de la sección 5.7; en particular, alcanzará un F1-macro no inferior a 0.85 y un recall-macro no inferior a 0.90 sobre el conjunto de prueba particionado por local, con una degradación no superior a 2 puntos de F1-macro respecto de su versión en float32, una latencia mediana no superior a 2 000 ms y un tamaño de modelo no superior a 20 MB sobre el Honor X6b. El modelo de línea base ResNet-50 no superará de forma significativa a la mejor arquitectura eficiente en F1-macro, entendiendo por diferencia significativa aquella cuyo intervalo de confianza bootstrap al 95 % (con remuestreo por local) excluye el cero, e incumplirá la restricción de tamaño, de modo que no constituirá una configuración desplegable.*

Esta expectativa se apoya en dos evidencias previas. En ImageNet, ResNet-50 alcanza una exactitud top-1 de 76,1 %, frente a 75,2 % de MobileNetV3-Large y 77,1 % de EfficientNet-B0, según los resultados reportados por He y colaboradores \[1\], Howard y colaboradores \[4\] y Tan y Le \[5\], de modo que la literatura no respalda una ventaja sistemática de la línea base. En cuanto al tamaño, sus cerca de 25,6 millones de parámetros ocupan alrededor de 98 MB en float32 y 25 MB en int8, por encima del límite de 20 MB. Este segundo componente se deriva del diseño más que de la experimentación: el límite responde a la distribución de actualizaciones con datos móviles y se fijó con independencia de este resultado. La parte informativa de la hipótesis reside por tanto en la comparación de desempeño predictivo y en la magnitud de las diferencias de latencia y memoria, que se miden en el dispositivo. La hipótesis es refutable en cualquiera de sus componentes. Si ninguna configuración eficiente alcanza los umbrales predictivos, si la degradación por cuantización excede el margen fijado, o si la línea base supera de forma significativa a las arquitecturas eficientes, el resultado conserva valor informativo y constituye evidencia sobre los límites del enfoque bajo estas restricciones.

### 5.9 Evaluación de la verificación contra convenio
La capacidad de detectar discrepancias se evalúa de forma separada del desempeño de clasificación, y su diseño reproduce la estructura del error que el proceso admite. Como el formulario despliega únicamente los muebles contratados para el local seleccionado, el mercaderista solo puede equivocarse eligiendo una fila distinta dentro de esa misma lista. Los casos de discrepancia, que el sistema debe detectar, se construyen en consecuencia mediante intercambio dentro del local: para un punto de venta con dos o tres convenios, la fotografía verificada de uno de sus muebles se empareja con la declaración de otro mueble del mismo local, generando una discrepancia de etiqueta conocida y operativamente posible. El intercambio entre locales distintos queda descartado, porque produce discrepancias que el formulario impide.

Sobre ese conjunto se reportan la tasa de detección de discrepancias y la tasa de falsas alarmas sobre los pares correctos, en los que la fotografía coincide con el mueble declarado, desagregadas por par de clases intercambiadas, lo que permite identificar qué combinaciones de muebles resultan confundibles en la práctica. La evaluación se realiza dos veces: sobre pares construidos a partir del corpus histórico etiquetado, que aporta volumen, y sobre las sumisiones reales de ODK acumuladas durante el proyecto, que aportan condiciones de operación auténticas con un número reducido de casos.

El proceso de levantamiento en ODK es reciente y no acumula todavía casos confirmados de selección incorrecta, de modo que los positivos reales de discrepancia son inexistentes al momento de la planificación. Esta construcción mide por tanto la capacidad del sistema de detectar discrepancias inducidas, y su alcance se limita a ese escenario controlado. La validación con discrepancias reales queda planteada como trabajo futuro, condicionada a la acumulación de casos confirmados una vez que el sistema entre en uso.

## 6. Sumario de Contenidos
El documento final se presentará en formato de congreso IEEE y se organizará de la siguiente manera:

- Introducción: contexto operativo del problema, definición formal de la tarea de clasificación, preguntas de investigación, objetivos y alcance.

- Estado del arte: reconocimiento visual en retail, arquitecturas eficientes para inferencia en el borde, técnicas de compresión de modelos, y limitación específica que el trabajo aborda.

- Descripción de la propuesta: taxonomía de clases, caracterización del conjunto de datos, estrategia de etiquetado (propagación débil y verificación manual), partición agrupada y diseño experimental.

- Desarrollo: entrenamiento de los tres modelos, cuantización post-entrenamiento, conversión para inferencia móvil e implementación del prototipo de verificación local.

- Experimentos y análisis de resultados: métricas predictivas por modelo y por clase, métricas de eficiencia medidas en dispositivo, curva de compromiso entre desempeño y costo, matriz de confusión y análisis de errores.

- Conclusiones y trabajo futuro: respuesta explícita a cada pregunta de investigación, configuración recomendada, limitaciones del diseño y extensiones posibles hacia detección de objetos y verificación de negocio y marca.

## 7. Recursos
### 7.1 Humanos
Estudiante responsable del proyecto, tutor académico, y personal del área comercial de Industrias Ales para validar la taxonomía de muebles, el protocolo de etiquetado y la utilidad operativa del prototipo.

### 7.2 Materiales
Computador personal para desarrollo y análisis; acceso a GPU local o en la nube para entrenamiento; repositorio de fotografías de punto de venta; maestra de convenios del canal moderno; herramienta de etiquetado; entorno Python con PyTorch o TensorFlow y librerías de análisis y visualización; dispositivo Honor X6b de la flota corporativa para las mediciones de eficiencia; y herramientas de despliegue móvil como TensorFlow Lite o un runtime equivalente.

### 7.3 Económicos
Se priorizarán herramientas de código abierto y recursos ya disponibles. El requerimiento principal corresponde a horas de GPU para los experimentos de entrenamiento y a almacenamiento para el corpus de imágenes. Al disponer de las imágenes, del dispositivo de referencia y de capacidad propia de etiquetado, no se prevé contratación de servicios externos. El presupuesto estimado es bajo.

## 8. Cronograma de Actividades
El cronograma cubre de septiembre a diciembre de 2026. Las actividades se solapan porque el etiquetado, el análisis de errores y el ajuste metodológico requieren iteraciones sucesivas. El etiquetado es la actividad de mayor riesgo del cronograma y se inicia de forma temprana.

| **Actividades**                                                                                                                                                           | **Sep** | **Oct** | **Nov** | **Dic** |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|---------|---------|
| A1: Ajuste del documento de planificación, taxonomía de clases y validación del alcance con el tutor.                                                                     | **X**   |         |         |         |
| A2: Revisión del estado del arte en visión para retail, modelos eficientes y compresión.                                                                                  | **X**   | **X**   |         |         |
| A3: Consolidación del corpus, verificación de ejemplares de las cinco clases, extracción de metadatos por OCR, reconciliación por coordenadas y deduplicación perceptual. | **X**   | **X**   |         |         |
| A4: Etiquetado y verificación manual (24 h semanales), con foco en clases minoritarias.                                                                                   | **X**   | **X**   | **X**   |         |
| A5: Entrenamiento de la línea base ResNet-50 y de los dos modelos eficientes.                                                                                             |         | **X**   | **X**   |         |
| A6: Evaluación comparativa, matriz de confusión y análisis de errores por clase.                                                                                          |         |         | **X**   | **X**   |
| A7: Cuantización post-entrenamiento, conversión a TFLite y medición en el Honor X6b.                                                                                      |         |         | **X**   | **X**   |
| A8: Prototipo Android de inferencia local y prueba de verificación contra convenio.                                                                                       |         |         | **X**   | **X**   |
| A9: Redacción final, artículo en formato IEEE y preparación de la defensa.                                                                                                |         |         |         | **X**   |

## 9. Referencias
\[1\] K. He, X. Zhang, S. Ren y J. Sun, “Deep Residual Learning for Image Recognition,” en Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

\[2\] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li y L. Fei-Fei, “ImageNet: A Large-Scale Hierarchical Image Database,” en Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255.

\[3\] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov y L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” en Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.

\[4\] A. Howard et al., “Searching for MobileNetV3,” en Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2019, pp. 1314–1324.

\[5\] M. Tan y Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” en Proc. 36th Int. Conf. Machine Learning (ICML), 2019, pp. 6105–6114.

\[6\] M. Tan y Q. V. Le, “EfficientNetV2: Smaller Models and Faster Training,” en Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 10096–10106.

\[7\] P. K. A. Vasu, J. Gabriel, J. Zhu, O. Tuzel y A. Ranjan, “MobileOne: An Improved One Millisecond Mobile Backbone,” en Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2023, pp. 7907–7917.

\[8\] S. Mehta y M. Rastegari, “MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer,” en Proc. Int. Conf. Learning Representations (ICLR), 2022.

\[9\] D. Qin et al., “MobileNetV4: Universal Models for the Mobile Ecosystem,” en Proc. European Conf. Computer Vision (ECCV), 2024.

\[10\] B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” en Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 2704–2713.

\[11\] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney y K. Keutzer, “A Survey of Quantization Methods for Efficient Neural Network Inference,” arXiv:2103.13630, 2021.

\[12\] G. Hinton, O. Vinyals y J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv:1503.02531, 2015.

\[13\] TensorFlow, “Post-training quantization,” TensorFlow Model Optimization, 2022.

\[14\] A. Tonioni, E. Serra y L. Di Stefano, “A Deep Learning Pipeline for Product Recognition on Store Shelves,” arXiv:1810.01733, 2018.

\[15\] X.-S. Wei, Q. Cui, L. Yang, P. Wang y L. Liu, “RPC: A Large-Scale Retail Product Checkout Dataset,” arXiv:1901.07249, 2019.

\[16\] E. Goldman, R. Herzig, A. Eisenschtat, J. Goldberger y T. Hassner, “Precise Detection in Densely Packed Scenes,” en Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5227–5236.

\[17\] J. Peng, C. Xiao y Y. Li, “RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification,” arXiv:2006.12634, 2020.

\[18\] Y. Wei, S. N. Tran, S. Xu, B. Kang y M. Springer, “Deep Learning for Retail Product Recognition: Challenges and Techniques,” Computational Intelligence and Neuroscience, vol. 2020, 2020.

\[19\] C. G. Northcutt, L. Jiang e I. L. Chuang, “Confident Learning: Estimating Uncertainty in Dataset Labels,” Journal of Artificial Intelligence Research, vol. 70, pp. 1373–1411, 2021.

\[20\] S. Kapoor y A. Narayanan, “Leakage and the Reproducibility Crisis in Machine-Learning-Based Science,” Patterns, vol. 4, no. 9, 2023.
