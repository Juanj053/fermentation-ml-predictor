"""
Pipelines de modelado predictivo, entrenamiento, validación cruzada y persistencia.
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SEED,
    CV_SPLITS,
    PARAM_GRID_GB,
    PARAM_GRID_RF,
    DEFAULT_MODEL_PATH,
    ALT_MODEL_PATH,
)


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Construye el catálogo de pipelines candidatos para el benchmark.
    Incluye un modelo lineal regularizado (Ridge) como baseline de comparación y
    ensambles no lineales (Random Forest y Gradient Boosting).
    """
    pipelines = {
        "Ridge_Linear_Baseline": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=SEED)),
            ]
        ),
        "Random_Forest": Pipeline(
            [
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Gradient_Boosting": Pipeline(
            [
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.08,
                        max_depth=5,
                        subsample=0.85,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    }
    return pipelines


def evaluate_model_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = CV_SPLITS,
    random_state: int = SEED,
) -> Dict[str, float]:
    """
    Ejecuta validación cruzada K-Fold de 5 particiones y calcula métricas clave.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        "cv_r2_mean": float(np.mean(cv_results["test_r2"])),
        "cv_r2_std": float(np.std(cv_results["test_r2"])),
        "cv_rmse_mean": float(-np.mean(cv_results["test_neg_rmse"])),
        "cv_rmse_std": float(np.std(cv_results["test_neg_rmse"])),
        "cv_mae_mean": float(-np.mean(cv_results["test_neg_mae"])),
        "cv_mae_std": float(np.std(cv_results["test_neg_mae"])),
    }


def train_and_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparameters: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Entrena y compara todos los pipelines candidatos mediante Validación Cruzada y conjunto de prueba.
    Selecciona el modelo de mejor desempeño bioingenieril.

    Returns
    -------
    best_bundle : dict
        Diccionario empaquetado con el pipeline ganador, sus metadatos y métricas.
    benchmark_results : dict
        Resultados comparativos de todos los modelos evaluados.
    """
    pipelines = build_candidate_pipelines()
    benchmark_results: Dict[str, Dict[str, Any]] = {}

    best_r2 = -float("inf")
    best_name = ""
    best_fitted_pipeline = None
    best_cv_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}
    best_params: Dict[str, Any] = {}

    for name, pipe in pipelines.items():
        # Validación Cruzada K-Fold en conjunto de entrenamiento
        cv_metrics = evaluate_model_cv(pipe, X_train, y_train)

        # Ajuste o afinamiento de hiperparámetros
        fitted_pipe = pipe
        current_params = {}
        if tune_hyperparameters and name == "Gradient_Boosting":
            grid = GridSearchCV(
                pipe,
                param_grid=PARAM_GRID_GB,
                cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
                scoring="r2",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            fitted_pipe = grid.best_estimator_
            current_params = grid.best_params_
        elif tune_hyperparameters and name == "Random_Forest":
            grid = GridSearchCV(
                pipe,
                param_grid=PARAM_GRID_RF,
                cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
                scoring="r2",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            fitted_pipe = grid.best_estimator_
            current_params = grid.best_params_
        else:
            fitted_pipe.fit(X_train, y_train)

        # Evaluación en conjunto de prueba independiente (Hold-out Test Set)
        y_pred = fitted_pipe.predict(X_test)
        test_r2 = float(r2_score(y_test, y_pred))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_mape = float(np.mean(np.abs((y_test.values - y_pred) / np.maximum(y_test.values, 1.0))) * 100.0)

        benchmark_results[name] = {
            "cv_r2_mean": cv_metrics["cv_r2_mean"],
            "cv_r2_std": cv_metrics["cv_r2_std"],
            "cv_rmse_mean": cv_metrics["cv_rmse_mean"],
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_mape": test_mape,
            "tuned": tune_hyperparameters and name in ["Gradient_Boosting", "Random_Forest"],
        }

        # Criterio de selección: Mayor R2 en Test
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_name = name
            best_fitted_pipeline = fitted_pipe
            best_cv_metrics = cv_metrics
            best_test_metrics = {
                "r2": test_r2,
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
            }
            best_params = current_params or fitted_pipe.get_params()

    # Empaquetado completo del modelo y artefactos
    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else FEATURE_COLUMNS
    best_bundle = {
        "pipeline": best_fitted_pipeline,
        "model_name": best_name,
        "feature_names": feature_names,
        "target_name": TARGET_COLUMN,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "cv_metrics": best_cv_metrics,
        "test_metrics": best_test_metrics,
        "best_params": {k: str(v) for k, v in best_params.items() if not k.startswith("regressor__") or not callable(v)},
        "benchmark_summary": benchmark_results,
    }

    return best_bundle, benchmark_results


def save_model_artifact(
    model_bundle: Dict[str, Any],
    filepath: Path,
    also_save_alt: bool = True,
) -> Path:
    """
    Persiste el diccionario de artefactos y pipeline serializado en formato joblib.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, filepath, compress=3)

    if also_save_alt and filepath == DEFAULT_MODEL_PATH:
        ALT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, ALT_MODEL_PATH, compress=3)

    return filepath


def load_model_artifact(filepath: Path) -> Dict[str, Any]:
    """
    Carga el artefacto serializado desde disco.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        # Intento con ruta alterna si existe
        if filepath == DEFAULT_MODEL_PATH and ALT_MODEL_PATH.exists():
            filepath = ALT_MODEL_PATH
        elif filepath == ALT_MODEL_PATH and DEFAULT_MODEL_PATH.exists():
            filepath = DEFAULT_MODEL_PATH
        else:
            raise FileNotFoundError(f"No se encontró el artefacto del modelo en: {filepath}")

    bundle = joblib.load(filepath)
    return bundle
