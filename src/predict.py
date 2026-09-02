"""
Motor de inferencia para predicciones de rendimiento individuales y por lotes (Batch).
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    DEFAULT_MODEL_PATH,
)
from src.model_pipeline import load_model_artifact


def assess_operational_regime(inputs: Dict[str, float]) -> Dict[str, Any]:
    """
    Evalúa el régimen operacional de las variables de entrada según las ventanas biofísicas.
    Genera advertencias operativas si los parámetros escapan de la zona segura.
    """
    warnings: List[str] = []
    regimes: List[str] = []

    temp = inputs.get("temperature_c", 31.0)
    ph = inputs.get("ph_level", 5.8)
    brix = inputs.get("initial_brix", 18.0)
    do = inputs.get("dissolved_oxygen_pct", 50.0)
    rpm = inputs.get("agitation_rpm", 400.0)

    # 1. Chequeo de Temperatura
    if temp > 38.0:
        warnings.append(f"Temperatura crítica ({temp:.1f}°C): Desnaturalización térmica irreversible.")
        regimes.append("Estrés Térmico Severo")
    elif temp < 25.0:
        warnings.append(f"Temperatura baja ({temp:.1f}°C): Cinética metabólica deprimida.")
        regimes.append("Baja Actividad Metabólica")

    # 2. Chequeo de pH
    if ph < 4.5 or ph > 7.5:
        warnings.append(f"pH desfavorable ({ph:.2f}): Desestabilización de membrana e ionización enzimática.")
        regimes.append("Desviación Crítica de pH")

    # 3. Chequeo de Azúcares (°Brix)
    if brix > 23.0:
        warnings.append(f"Azúcar excesivo ({brix:.1f}°Brix): Efecto Haldane / estrés osmótico.")
        regimes.append("Inhibición por Sustrato")
    elif brix < 12.0:
        warnings.append(f"Sustrato insuficiente ({brix:.1f}°Brix): Agotamiento temprano de fuente de carbono.")
        regimes.append("Limitación de Carbono")

    # 4. Chequeo de Oxígeno y Agitación
    if do < 20.0:
        warnings.append(f"Hipoxia ({do:.1f}% DO): Limitación de transferencia de O2.")
        regimes.append("Régimen Anaerobio/Hipóxico")

    if rpm > 550.0:
        warnings.append(f"Agitación violenta ({rpm:.0f} RPM): Alto esfuerzo cortante (Shear Stress).")
        regimes.append("Cizallamiento Hidrodinámico")

    overall_status = "Condición Óptima" if not regimes else ", ".join(regimes)
    return {
        "regime": overall_status,
        "warnings": warnings,
        "is_safe": len(warnings) == 0,
    }


def predict_single(
    model_bundle: Union[Dict[str, Any], Path, str],
    input_data: Dict[str, float],
) -> Dict[str, Any]:
    """
    Realiza la predicción de rendimiento final (g/L) para un lote individual.

    Parameters
    ----------
    model_bundle : dict or Path
        Artefacto del modelo cargado o ruta al archivo joblib.
    input_data : dict
        Diccionario con las 8 variables de entrada requeridas.

    Returns
    -------
    dict
        Resultado estructurado con rendimiento estimado, intervalo de confianza y diagnóstico.
    """
    if isinstance(model_bundle, (str, Path)):
        model_bundle = load_model_artifact(Path(model_bundle))

    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle.get("feature_names", FEATURE_COLUMNS)

    # Validar presencia de características requeridas
    missing = [fn for fn in feature_names if fn not in input_data]
    if missing:
        raise ValueError(f"Faltan las siguientes variables de entrada requeridas: {missing}")

    # Construir DataFrame de una fila
    row_values = [input_data[fn] for fn in feature_names]
    df_single = pd.DataFrame([row_values], columns=feature_names)

    # Inferencia
    predicted_yield = float(pipeline.predict(df_single)[0])
    predicted_yield = float(np.clip(predicted_yield, 0.0, 95.0))

    # Diagnóstico biofísico
    diagnosis = assess_operational_regime(input_data)

    # Estimación de intervalo de confianza basado en RMSE del modelo
    test_rmse = model_bundle.get("test_metrics", {}).get("rmse", 2.5)
    ci_lower = max(0.0, predicted_yield - 1.96 * test_rmse)
    ci_upper = min(95.0, predicted_yield + 1.96 * test_rmse)

    return {
        "predicted_yield_g_l": round(predicted_yield, 2),
        "ci_95_lower_g_l": round(ci_lower, 2),
        "ci_95_upper_g_l": round(ci_upper, 2),
        "model_used": model_bundle.get("model_name", "TrainedModel"),
        "operational_regime": diagnosis["regime"],
        "is_safe": diagnosis["is_safe"],
        "warnings": diagnosis["warnings"],
        "input_features": {fn: input_data[fn] for fn in feature_names},
    }


def predict_batch(
    model_bundle: Union[Dict[str, Any], Path, str],
    input_csv_path: Union[str, Path],
    output_csv_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Realiza inferencia sobre un conjunto de lotes en archivo CSV y exporta resultados enriquecidos.
    """
    if isinstance(model_bundle, (str, Path)):
        model_bundle = load_model_artifact(Path(model_bundle))

    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_csv_path}")

    df_in = pd.read_csv(input_csv_path)
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle.get("feature_names", FEATURE_COLUMNS)

    # Validar columnas
    for fn in feature_names:
        if fn not in df_in.columns:
            raise ValueError(f"Columna requerida '{fn}' no encontrada en el CSV de entrada.")

    X = df_in[feature_names]
    predictions = pipeline.predict(X)
    predictions = np.clip(predictions, 0.0, 95.0)

    df_out = df_in.copy()
    df_out["predicted_yield_g_l"] = np.round(predictions, 2)

    # Diagnósticos individuales
    regimes = []
    warnings_list = []
    for _, row in X.iterrows():
        diag = assess_operational_regime(row.to_dict())
        regimes.append(diag["regime"])
        warnings_list.append("; ".join(diag["warnings"]) if diag["warnings"] else "OK")

    df_out["operational_regime"] = regimes
    df_out["operational_warnings"] = warnings_list

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    return df_out
