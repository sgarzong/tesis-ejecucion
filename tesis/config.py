"""Constantes del protocolo experimental (planificación v4, secciones 2.2, 5.4, 5.5 y 5.7)."""

CLASSES = [
    "chiclera",
    "lateral",
    "check_out",
    "cabecera",
    "exhibidor_piso",
    "captura_invalida",
    "contenido_ajeno",
]
REJECT = {"captura_invalida", "contenido_ajeno"}

# Clases del ResNet-18 de explicabilidad (preetiquetado) -> CLASSES. ISLA_RUMA_BIOMBO = ruma.
# GANCHERA no está en la taxonomía (sección 2.2): se sugiere como contenido ajeno y el revisor decide.
TO_TESIS = {"CABECERA_FINAL": "cabecera", "CHECK": "check_out", "CHICLERA": "chiclera",
            "ISLA_RUMA_BIOMBO": "exhibidor_piso", "LATERAL": "lateral", "GANCHERA": "contenido_ajeno"}

# Nombre del mueble en el convenio o declarado en ODK -> clase. Tabla de la sección 2.2 (RUMA = exhibidor de piso),
# más las nomenclaturas de acuerdos comerciales y extravisibilidad: el arete es un lateral de cabecera; mueble gris,
# four way y biombo son exhibidores de piso. Lo que no está aquí (percha, OTA, exhibidor de counter, ganchera, mueble
# de frutas y verduras) no es de la taxonomía: contenido ajeno, como GANCHERA en TO_TESIS, y el revisor decide.
CLASE_DECLARADA = {
    **dict.fromkeys(["CHICLERA", "CHICLERAS"], "chiclera"),
    **dict.fromkeys(["LATERAL SIMPLE", "LATEAL SIMPLE", "LATERAL DOBLE", "LATERAL DE CABECERA",
                     "CHECK OUT LATERAL DOBLE", "ARETE"], "lateral"),
    **dict.fromkeys(["CHECK OUT", "CHECK OUT SIMPLE", "CAJA"], "check_out"),
    **dict.fromkeys(["CABECERA", "CABECERA FRONTAL", "CABECERA SECA", "CABECERA FASTLINE", "MUEBLE GRIS/CABECERA",
                     "BANDEJA DE CABECERA"], "cabecera"),
    **dict.fromkeys(["EXHIBIDOR DE PISO", "RUMA", "RUMAS", "RUMA COMPARTIDA", "MEDIA RUMA", "FOUR WAY", "FOUR WAYS",
                     "MUEBLE GRIS", "MUBLE GRIS", "MUEBLE GRIS COMPARTIDO", "BIOMBO", "BANDEJA DE BIOMBO",
                     "ISLA", "ISLA BOTELLA"],  # la isla va con la ruma, como ISLA_RUMA_BIOMBO en TO_TESIS
                    "exhibidor_piso"),
}

MODELS = ["resnet50", "mobilenetv3", "efficientnetb0"]
IMG_SIZE = 224
SEED = 42

# Texto de Timestamp Camera (abajo a la derecha, en fotos verticales y apaisadas), como fracción de ancho y alto.
# Se enmascara porque su última línea es el nombre del local, que delata la clase vía el convenio (también en
# ODK, que usa la misma app). Calibrado midiendo el texto en 777 fotos (400 verticales, 377 apaisadas): cubre
# hasta 0.27/0.33 de alto (p90 vertical/apaisada) y 0.70 de ancho. Lo que asoma en el 10 % restante son las
# líneas de fecha (arriba); el nombre del local siempre queda dentro.
MASK_W, MASK_H = 0.70, 0.33
# Logo de la app (arriba a la derecha, p90 0.19 x 0.17). No se enmascara: aparece igual en todas las fotos de
# ambos corpus y no informa la clase. tesis.degrade usa la zona para restaurarlo nítido en las sintéticas.
LOGO_W, LOGO_H = 0.20, 0.20

# Tope de imágenes por local en entrenamiento (riesgo 5, sección 5.2). Calibrar.
MAX_PER_STORE = 300

# Experimento B: entrenamiento hasta sep-2025, prueba oct-dic 2025.
TEMPORAL_CUTOFF = "2025-10-01"

# Días a cada lado de la fecha de una maestra de convenios en que rige las fotos (tesis.convenios): los convenios
# cambian, y una maestra de junio no dice qué tenía el local un año antes.
VENTANA_CONVENIO = 183

THRESHOLDS = {
    "f1_macro": 0.85,
    "recall_macro": 0.90,
    "reject_recall": 0.90,
    "false_alarm": 0.05,
    "quant_drop": 0.02,
    "lat_p50_ms": 2000,
    "lat_p95_ms": 3000,
    "mem_mb": 300,
    "size_mb": 20,
}
