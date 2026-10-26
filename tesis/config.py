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
