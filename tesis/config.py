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

MODELS = ["resnet50", "mobilenetv3", "efficientnetb0"]
IMG_SIZE = 224
SEED = 42

# Superposición de Timestamp Camera (esquina inferior derecha), como fracción de ancho y alto.
# Valor de calibración: ajustar mirando muestras reales de ambos corpus.
MASK_W, MASK_H = 0.55, 0.22

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
