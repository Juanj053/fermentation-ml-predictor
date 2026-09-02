"""
Generador de datos sintéticos basados en cinética biofísica de fermentación.
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_METADATA,
    KINETICS_PARAMS,
    SEED,
    DEFAULT_DATA_PATH,
    ALT_DATA_PATH,
)


def calculate_temperature_factor(temperature_c: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Calcula el factor de respuesta térmica f_T(T) mediante modelo Arrhenius-desnaturalización.
    Presenta un óptimo a 31.0°C y penalización exponencial severa por encima de 38.0°C.
    """
    t_opt = params["temp_opt"]
    t_w = params["temp_width"]
    t_denat = params["temp_denat_thresh"]
    k_denat = params["temp_denat_k"]

    # Respuesta gaussiana en torno al óptimo
    f_t = np.exp(-((temperature_c - t_opt) ** 2) / (2.0 * (t_w ** 2)))

    # Penalización por desnaturalización proteica a altas temperaturas
    denat_mask = temperature_c > t_denat
    f_t[denat_mask] *= np.exp(-k_denat * (temperature_c[denat_mask] - t_denat))
    return np.clip(f_t, 0.0, 1.0)


def calculate_ph_factor(ph_level: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Calcula el factor de ionización enzimática f_pH(pH) basado en la cinética de Dixon-Webb.
    Óptimo en pH 5.8 con decaimiento simétrico por alteración del estado de carga del sitio activo.
    """
    ph_opt = params["ph_opt"]
    ph_w = params["ph_width"]
    f_ph = np.exp(-((ph_level - ph_opt) ** 2) / (2.0 * (ph_w ** 2)))
    return np.clip(f_ph, 0.0, 1.0)


def calculate_substrate_factor(initial_brix: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Calcula el factor de consumo de sustrato f_S(Brix) mediante cinética de Haldane-Andrews.
    Captura saturación tipo Monod a bajas concentraciones e inhibición osmótica/catabólica a >22°Brix.
    """
    ks = params["brix_ks"]
    ki = params["brix_ki"]
    scale = params["brix_scale"]
    
    # Ecuación de Haldane: f_s = scale * S / (Ks + S + S^2 / Ki)
    f_s = scale * (initial_brix / (ks + initial_brix + ((initial_brix ** 2) / ki)))
    return np.clip(f_s, 0.0, 1.0)


def calculate_oxygen_transfer_factor(
    dissolved_oxygen_pct: np.ndarray,
    agitation_rpm: np.ndarray,
    aeration_vvm: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """
    Calcula el factor de suministro de oxígeno y esfuerzo cortante f_O2(DO, RPM, vvm).
    Modela el coeficiente volumétrico de transferencia de masa (kLa) y la penalización
    por cizallamiento hidrodinámico cuando la velocidad de agitación supera 550 RPM.
    """
    rpm_ref = params["rpm_ref"]
    do_ref = params["do_ref"]
    vvm_ref = params["vvm_ref"]
    shear_thresh = params["shear_thresh"]
    shear_scale = params["shear_scale"]

    # Potencia combinada de kLa
    kappa_o2 = (
        ((agitation_rpm / rpm_ref) ** 0.8)
        * ((dissolved_oxygen_pct / do_ref) ** 0.5)
        * ((aeration_vvm / vvm_ref) ** 0.3)
    )

    # Saturación de transferencia de oxígeno
    transfer_term = np.minimum(1.0, kappa_o2 / 1.1)

    # Factor de daño por cizallamiento (shear stress damage a células)
    shear_excess = np.maximum(0.0, (agitation_rpm - shear_thresh) / shear_scale)
    shear_penalty = np.clip(1.0 - shear_excess, 0.2, 1.0)

    f_o2 = transfer_term * shear_penalty
    return np.clip(f_o2, 0.0, 1.0)


def calculate_time_factor(fermentation_time_h: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Calcula la dinámica de acumulación de metabolito secundario/biomasa f_t(t)
    mediante una curva sigmoidal logística con meseta asintótica.
    """
    k = params["time_k"]
    t_mid = params["time_mid"]
    f_t = 1.0 / (1.0 + np.exp(-k * (fermentation_time_h - t_mid)))
    return np.clip(f_t, 0.0, 1.0)


def calculate_inoculum_factor(inoculum_od600: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Calcula el factor de fase de latencia y vigor del inóculo f_inoc(OD600).
    """
    base = params["inoc_base"]
    scale = params["inoc_scale"]
    k = params["inoc_k"]
    f_inoc = base + scale * (1.0 - np.exp(-k * inoculum_od600))
    return np.clip(f_inoc, 0.0, 1.0)


def generate_bioprocess_data(
    n_samples: int = 1500,
    random_state: int = SEED,
    inject_anomalies: bool = True,
    params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético biológicamente fundamentado para fermentación en biorreactores.

    Parameters
    ----------
    n_samples : int
        Número total de lotes de fermentación a generar (por defecto 1500).
    random_state : int
        Semilla para reproducibilidad estocástica.
    inject_anomalies : bool
        Si es True, inyecta un 2% de lotes anómalos (fallas operativas o contaminación).
    params : dict, optional
        Diccionario de parámetros cinéticos de sobreescritura.

    Returns
    -------
    pd.DataFrame
        DataFrame con las 8 variables operativas de entrada y la variable objetivo 'yield_g_l'.
    """
    rng = np.random.default_rng(random_state)
    kin_params = KINETICS_PARAMS.copy()
    if params:
        kin_params.update(params)

    # 1. Generación de variables operacionales dentro de rangos industriales plausibles
    # Usamos una combinación de distribuciones normales truncadas y uniformes para cubrir
    # tanto el espacio operativo nominal como regiones de exploración y perturbación.
    
    # Temperatura: Media en 31.5°C con dispersión entre 22.0 y 40.0°C
    temp = rng.normal(loc=31.2, scale=3.8, size=n_samples)
    temp = np.clip(temp, FEATURE_METADATA["temperature_c"]["min"], FEATURE_METADATA["temperature_c"]["max"])

    # pH: Centrado en 5.85 con variación entre 4.0 y 8.0
    ph = rng.normal(loc=5.85, scale=0.8, size=n_samples)
    ph = np.clip(ph, FEATURE_METADATA["ph_level"]["min"], FEATURE_METADATA["ph_level"]["max"])

    # Grados Brix iniciales: 10.0 a 26.0 °Brix
    brix = rng.uniform(
        low=FEATURE_METADATA["initial_brix"]["min"],
        high=FEATURE_METADATA["initial_brix"]["max"],
        size=n_samples,
    )

    # Oxígeno Disuelto: 10.0 a 90.0 %
    do = rng.uniform(
        low=FEATURE_METADATA["dissolved_oxygen_pct"]["min"],
        high=FEATURE_METADATA["dissolved_oxygen_pct"]["max"],
        size=n_samples,
    )

    # Agitación: 150 a 600 RPM con sesgo operacional en 380-450 RPM
    rpm = rng.normal(loc=400.0, scale=80.0, size=n_samples)
    rpm = np.clip(rpm, FEATURE_METADATA["agitation_rpm"]["min"], FEATURE_METADATA["agitation_rpm"]["max"])

    # Tiempo de fermentación: 18.0 a 96.0 h (distribución multimodal de lotes cortos y largos)
    time_h = rng.uniform(
        low=FEATURE_METADATA["fermentation_time_h"]["min"],
        high=FEATURE_METADATA["fermentation_time_h"]["max"],
        size=n_samples,
    )

    # Inóculo OD600: 0.5 a 3.0
    inoc = rng.normal(loc=1.8, scale=0.5, size=n_samples)
    inoc = np.clip(inoc, FEATURE_METADATA["inoculum_od600"]["min"], FEATURE_METADATA["inoculum_od600"]["max"])

    # Aireación: 0.2 a 2.0 vvm
    vvm = rng.normal(loc=1.05, scale=0.35, size=n_samples)
    vvm = np.clip(vvm, FEATURE_METADATA["aeration_vvm"]["min"], FEATURE_METADATA["aeration_vvm"]["max"])

    # 2. Evaluación de las funciones cinéticas parciales
    f_t = calculate_temperature_factor(temp, kin_params)
    f_ph = calculate_ph_factor(ph, kin_params)
    f_s = calculate_substrate_factor(brix, kin_params)
    f_o2 = calculate_oxygen_transfer_factor(do, rpm, vvm, kin_params)
    f_time = calculate_time_factor(time_h, kin_params)
    f_inoc = calculate_inoculum_factor(inoc, kin_params)

    # 3. Interacción cinética multiplicativa y rendimiento base
    y_max = kin_params["y_max"]
    yield_clean = y_max * f_t * f_ph * f_s * f_o2 * f_time * f_inoc

    # 4. Inyección de ruido experimental estocástico (variabilidad de bioproceso real)
    noise = rng.normal(loc=0.0, scale=kin_params["noise_sigma"], size=n_samples)
    final_yield = yield_clean + noise

    # 5. Inyección de fallas anómalas de lote (contaminación bacteriófaga o fallo de agitador)
    is_anomaly = np.zeros(n_samples, dtype=bool)
    if inject_anomalies and kin_params["anomaly_rate"] > 0:
        n_anomalies = int(n_samples * kin_params["anomaly_rate"])
        if n_anomalies > 0:
            anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
            final_yield[anomaly_idx] *= kin_params["anomaly_factor"]
            is_anomaly[anomaly_idx] = True

    # 6. Acotación a límites biofísicos realistas [0, 95.0 g/L]
    final_yield = np.clip(final_yield, 0.0, 95.0)

    # 7. Construcción del DataFrame estructurado
    batch_ids = [f"BATCH-{i+1:05d}" for i in range(n_samples)]
    df = pd.DataFrame(
        {
            "batch_id": batch_ids,
            "temperature_c": np.round(temp, 2),
            "ph_level": np.round(ph, 3),
            "initial_brix": np.round(brix, 2),
            "dissolved_oxygen_pct": np.round(do, 2),
            "agitation_rpm": np.round(rpm, 1),
            "fermentation_time_h": np.round(time_h, 2),
            "inoculum_od600": np.round(inoc, 3),
            "aeration_vvm": np.round(vvm, 3),
            "is_anomaly": is_anomaly,
            TARGET_COLUMN: np.round(final_yield, 3),
        }
    )

    return df


def save_dataset(df: pd.DataFrame, filepath: Path, also_save_alt: bool = True) -> Path:
    """
    Guarda el DataFrame generado en formato CSV creando directorios si no existen.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    if also_save_alt and filepath == DEFAULT_DATA_PATH:
        ALT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ALT_DATA_PATH, index=False)

    return filepath


if __name__ == "__main__":
    df_sample = generate_bioprocess_data(1500)
    print(f"Dataset generado exitosamente con {len(df_sample)} filas.")
    print(df_sample.describe().T)
    out_path = save_dataset(df_sample, DEFAULT_DATA_PATH)
    print(f"Archivo guardado en: {out_path}")
