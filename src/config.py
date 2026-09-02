"""
Configuración global, constantes bioprocesales e hiperparámetros.
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)
"""

from pathlib import Path
from typing import Dict, List, Any

# Base Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# File Paths (Primary and Aliases for full compatibility)
DEFAULT_DATA_PATH = DATA_DIR / "synthetic_fermentation_data.csv"
ALT_DATA_PATH = DATA_DIR / "fermentation_batches.csv"

DEFAULT_MODEL_PATH = MODELS_DIR / "best_yield_predictor.joblib"
ALT_MODEL_PATH = MODELS_DIR / "yield_model.joblib"

DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"
ALT_METRICS_PATH = REPORTS_DIR / "metrics_summary.json"

DEFAULT_PLOT_PATH = REPORTS_DIR / "evaluation_plots.png"

# Feature and Target Column Specifications
FEATURE_COLUMNS: List[str] = [
    "temperature_c",
    "ph_level",
    "initial_brix",
    "dissolved_oxygen_pct",
    "agitation_rpm",
    "fermentation_time_h",
    "inoculum_od600",
    "aeration_vvm",
]

TARGET_COLUMN: str = "yield_g_l"

# Operational Variables Metadata & Bounds
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "temperature_c": {
        "name": "Temperatura",
        "unit": "°C",
        "min": 22.0,
        "max": 40.0,
        "nominal_min": 30.0,
        "nominal_max": 32.0,
        "optimal": 31.0,
        "description": "Temperatura de cultivo en el biorreactor.",
    },
    "ph_level": {
        "name": "pH",
        "unit": "pH",
        "min": 4.0,
        "max": 8.0,
        "nominal_min": 5.5,
        "nominal_max": 6.2,
        "optimal": 5.8,
        "description": "Nivel de acidez/alcalinidad del caldo de fermentación.",
    },
    "initial_brix": {
        "name": "Azúcares Iniciales (°Brix)",
        "unit": "°Brix",
        "min": 10.0,
        "max": 26.0,
        "nominal_min": 16.0,
        "nominal_max": 20.0,
        "optimal": 18.0,
        "description": "Concentración inicial de azúcares fermentables.",
    },
    "dissolved_oxygen_pct": {
        "name": "Oxígeno Disuelto",
        "unit": "%",
        "min": 10.0,
        "max": 90.0,
        "nominal_min": 30.0,
        "nominal_max": 60.0,
        "optimal": 50.0,
        "description": "Saturación de oxígeno disuelto en el medio.",
    },
    "agitation_rpm": {
        "name": "Velocidad de Agitación",
        "unit": "RPM",
        "min": 150.0,
        "max": 600.0,
        "nominal_min": 350.0,
        "nominal_max": 450.0,
        "optimal": 400.0,
        "description": "Velocidad de giro del impelente (transferencia de masa y cizalla).",
    },
    "fermentation_time_h": {
        "name": "Tiempo de Fermentación",
        "unit": "h",
        "min": 18.0,
        "max": 96.0,
        "nominal_min": 48.0,
        "nominal_max": 72.0,
        "optimal": 60.0,
        "description": "Duración total de la corrida del lote.",
    },
    "inoculum_od600": {
        "name": "Densidad de Inóculo (OD600)",
        "unit": "OD600",
        "min": 0.5,
        "max": 3.0,
        "nominal_min": 1.5,
        "nominal_max": 2.2,
        "optimal": 2.0,
        "description": "Concentración óptica inicial de biomasa inoculada.",
    },
    "aeration_vvm": {
        "name": "Tasa de Aireación",
        "unit": "vvm",
        "min": 0.2,
        "max": 2.0,
        "nominal_min": 0.8,
        "nominal_max": 1.2,
        "optimal": 1.0,
        "description": "Flujo volumétrico de aire por volumen de medio por minuto.",
    },
}

# Biophysical Kinetics Parameters (Data Generating Process)
KINETICS_PARAMS: Dict[str, float] = {
    "y_max": 88.0,            # Rendimiento máximo teórico (g/L)
    "temp_opt": 31.0,         # Temperatura óptima (°C)
    "temp_width": 3.5,        # Ancho de campana térmica (°C)
    "temp_denat_thresh": 38.0,# Umbral de desnaturalización térmica (°C)
    "temp_denat_k": 0.8,      # Tasa de caída térmica por desnaturalización
    "ph_opt": 5.8,            # pH óptimo (Dixon-Webb)
    "ph_width": 0.75,         # Ancho de campana de pH
    "brix_ks": 3.0,           # Constante de afinidad de sustrato (°Brix)
    "brix_ki": 35.0,          # Constante de inhibición por sustrato (°Brix)
    "brix_scale": 2.2,        # Factor de escala Haldane
    "rpm_ref": 400.0,         # RPM de referencia para kLa
    "do_ref": 50.0,           # DO% de referencia
    "vvm_ref": 1.0,           # vvm de referencia
    "shear_thresh": 550.0,    # Umbral de daño por esfuerzo cortante
    "shear_scale": 300.0,     # Factor de sensibilidad de cizalla
    "time_k": 0.085,          # Constante de acumulación logística de producto
    "time_mid": 38.0,         # Punto de inflexión temporal (h)
    "inoc_base": 0.75,        # Rendimiento base con inóculo mínimo
    "inoc_scale": 0.25,       # Factor modulador por inóculo
    "inoc_k": 1.2,            # Constante de saturación de inóculo
    "noise_sigma": 1.8,       # Desviación estándar de ruido experimental (g/L)
    "anomaly_rate": 0.02,     # Tasa de anomalías por contaminación/falla mecánica (2%)
    "anomaly_factor": 0.25,   # Factor de reducción de rendimiento en lotes fallidos
}

# Machine Learning & Cross-Validation Settings
SEED: int = 42
DEFAULT_SAMPLE_COUNT: int = 1500
TEST_SIZE: float = 0.20
CV_SPLITS: int = 5

# Hyperparameter search grids
PARAM_GRID_GB = {
    "regressor__n_estimators": [100, 150, 200],
    "regressor__learning_rate": [0.03, 0.08, 0.12],
    "regressor__max_depth": [3, 4, 5],
    "regressor__subsample": [0.8, 0.9, 1.0],
}

PARAM_GRID_RF = {
    "regressor__n_estimators": [100, 150, 200],
    "regressor__max_depth": [8, 12, 16, None],
    "regressor__min_samples_split": [2, 4, 8],
    "regressor__min_samples_leaf": [1, 2],
}
