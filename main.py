#!/usr/bin/env python3
"""
Punto de entrada principal y CLI para el Predictor de Rendimiento de Fermentación (ML).
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)

Uso:
  python3 main.py                                      # Ejecución completa de punta a punta (Zero-config)
  python3 main.py --train --samples 2000               # Generación y entrenamiento personalizado
  python3 main.py --evaluate                           # Evaluación y generación de gráficos diagnósticos
  python3 main.py --predict --temp 31.0 --ph 5.8       # Inferencia para un lote individual
  python3 main.py --predict-csv data/new_batches.csv   # Inferencia por lotes
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SEED,
    DEFAULT_SAMPLE_COUNT,
    TEST_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_PLOT_PATH,
    REPORTS_DIR,
    FIGURES_DIR,
    FEATURE_METADATA,
)
from src.data_generator import generate_bioprocess_data, save_dataset
from src.model_pipeline import (
    train_and_benchmark,
    save_model_artifact,
    load_model_artifact,
)
from src.evaluate import (
    compute_metrics_dict,
    save_metrics_report,
    generate_diagnostic_suite,
)
from src.predict import predict_single, predict_batch


def print_banner():
    print("=" * 78)
    print(" 🌿 FERMENTATION YIELD PREDICTOR — MACHINE LEARNING & BIOPROCESS")
    print("    Modelado cinético biofísico y optimización de bioprocesos industriales")
    print("=" * 78)


def run_full_pipeline(
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
    output_dir: Path = REPORTS_DIR,
    tune: bool = False,
) -> int:
    """
    Ejecuta el flujo completo de punta a punta:
    1. Generación de datos sintéticos biofísicos.
    2. Entrenamiento y benchmark de modelos (Ridge, RF, GB) con CV de 5 pliegues.
    3. Persistencia del mejor artefacto.
    4. Evaluación rigurosa en conjunto de prueba independiente.
    5. Exportación de métricas JSON y generación de gráficos diagnósticos (4 paneles, 300 DPI).
    6. Prueba de inferencia demostrativa.
    """
    print("\n[FASE 1/5] Generando conjunto de datos con cinética biofísica...")
    df = generate_bioprocess_data(n_samples=n_samples, random_state=SEED)
    saved_data = save_dataset(df, data_path)
    print(f"  ✓ {len(df)} lotes generados y persistidos en: {saved_data}")
    print(f"  ✓ Rendimiento promedio: {df[TARGET_COLUMN].mean():.2f} ± {df[TARGET_COLUMN].std():.2f} g/L (Rango: {df[TARGET_COLUMN].min():.1f} – {df[TARGET_COLUMN].max():.1f} g/L)")

    print("\n[FASE 2/5] Particionando datos y entrenando modelos candidatos...")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
    print(f"  ✓ Partición: {len(X_train)} entrenamiento ({100*(1-TEST_SIZE):.0f}%), {len(X_test)} prueba ({100*TEST_SIZE:.0f}%)")

    print("\n[FASE 3/5] Benchmark de algoritmos y validación cruzada (5-Fold CV)...")
    best_bundle, benchmark_summary = train_and_benchmark(
        X_train, y_train, X_test, y_test, tune_hyperparameters=tune
    )

    print("\n  " + "-" * 72)
    print(f"  {'Modelo':<24} | {'CV R² (Mean ± Std)':<20} | {'Test R²':<10} | {'Test RMSE (g/L)':<12}")
    print("  " + "-" * 72)
    for model_name, res in benchmark_summary.items():
        cv_str = f"{res['cv_r2_mean']:.3f} ± {res['cv_r2_std']:.3f}"
        print(f"  {model_name:<24} | {cv_str:<20} | {res['test_r2']:<10.3f} | {res['test_rmse']:<12.2f}")
    print("  " + "-" * 72)

    winning_model = best_bundle["model_name"]
    print(f"\n  🏆 Modelo Ganador Seleccionado: {winning_model}")
    saved_model = save_model_artifact(best_bundle, model_path)
    print(f"  ✓ Artefacto serializado guardado en: {saved_model}")

    print("\n[FASE 4/5] Evaluación diagnóstica y visualización bioingenieril...")
    y_pred = best_bundle["pipeline"].predict(X_test)
    metrics = compute_metrics_dict(y_test.values, y_pred)

    full_metrics_report = {
        "selected_model": winning_model,
        "n_samples_total": n_samples,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "test_metrics": metrics,
        "cv_metrics": best_bundle["cv_metrics"],
        "benchmark_summary": benchmark_summary,
        "timestamp_utc": best_bundle["training_timestamp"],
    }
    saved_metrics = save_metrics_report(full_metrics_report, metrics_path)
    print(f"  ✓ Métricas guardadas en: {saved_metrics}")
    print(f"    - R² Test:   {metrics['r2_score']:.4f}")
    print(f"    - RMSE:      {metrics['rmse_g_l']:.2f} g/L")
    print(f"    - MAE:       {metrics['mae_g_l']:.2f} g/L")
    print(f"    - MAPE:      {metrics['mape_percent']:.2f}%")
    print(f"    - Pearson r: {metrics['pearson_correlation']:.4f}")

    figs_dir = output_dir / "figures"
    composite_png, individual_pngs = generate_diagnostic_suite(
        best_bundle, X_test, y_test, output_composite_path=plot_path, figures_dir=figs_dir
    )
    print(f"  ✓ Panel diagnóstico maestro (300 DPI) generado en: {composite_png}")
    for fig_k, fig_p in individual_pngs.items():
        print(f"    • {fig_k}: {fig_p}")

    print("\n[FASE 5/5] Inferencia demostrativa en condiciones de operación nominal...")
    sample_nominal = {
        "temperature_c": 31.2,
        "ph_level": 5.85,
        "initial_brix": 18.5,
        "dissolved_oxygen_pct": 52.0,
        "agitation_rpm": 410.0,
        "fermentation_time_h": 60.0,
        "inoculum_od600": 2.0,
        "aeration_vvm": 1.1,
    }
    prediction = predict_single(best_bundle, sample_nominal)
    print(f"  • Condiciones de prueba: T={sample_nominal['temperature_c']}°C, pH={sample_nominal['ph_level']}, Brix={sample_nominal['initial_brix']}°Bx, DO={sample_nominal['dissolved_oxygen_pct']}%")
    print(f"  • Rendimiento Predicho:  {prediction['predicted_yield_g_l']:.2f} g/L (IC 95%: [{prediction['ci_95_lower_g_l']:.1f}, {prediction['ci_95_upper_g_l']:.1f}] g/L)")
    print(f"  • Estado Operativo:      {prediction['operational_regime']}")

    print("\n" + "=" * 78)
    print(" ✅ EJECUCIÓN COMPLETADA EXITOSAMENTE (Código de salida: 0)")
    print("=" * 78)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictor de Rendimiento de Fermentación mediante Machine Learning (Bioingeniería)."
    )
    parser.add_argument("--train", action="store_true", help="Entrena los modelos predictivos.")
    parser.add_argument("--evaluate", action="store_true", help="Evalúa el modelo y genera gráficos diagnósticos.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT, help="Número de lotes sintéticos a generar.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Ruta al archivo CSV de datos.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Ruta al archivo joblib del modelo.")
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR), help="Directorio para guardar reportes y figuras.")
    parser.add_argument("--tune", action="store_true", help="Ejecutar búsqueda exhaustiva de hiperparámetros (GridSearchCV).")
    parser.add_argument("--export-dataset", action="store_true", help="Solo generar y exportar el dataset sintético.")
    
    # Flags para inferencia
    parser.add_argument("--predict", action="store_true", help="Ejecuta inferencia para una muestra individual.")
    parser.add_argument("--predict-csv", type=str, default=None, help="Ruta al archivo CSV para inferencia por lotes.")
    parser.add_argument("--output-csv", type=str, default=None, help="Ruta para guardar predicciones por lotes.")
    
    # Argumentos individuales para predicción
    parser.add_argument("--temp", type=float, default=None, help="Temperatura en °C.")
    parser.add_argument("--ph", type=float, default=None, help="pH del medio.")
    parser.add_argument("--brix", type=float, default=None, help="Azúcares iniciales (°Brix).")
    parser.add_argument("--do", type=float, default=None, help="Oxígeno disuelto (% saturación).")
    parser.add_argument("--rpm", type=float, default=None, help="Velocidad de agitación (RPM).")
    parser.add_argument("--time", type=float, default=None, help="Tiempo de fermentación (h).")
    parser.add_argument("--inoculum", type=float, default=None, help="Densidad de inóculo (OD600).")
    parser.add_argument("--aeration", type=float, default=None, help="Tasa de aireación (vvm).")

    return parser.parse_args()


def main() -> int:
    print_banner()
    args = parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "evaluation_plots.png"

    # Caso 1: Solo exportar dataset
    if args.export_dataset:
        print(f"\nGenerando conjunto de datos con {args.samples} muestras...")
        df = generate_bioprocess_data(n_samples=args.samples, random_state=SEED)
        saved = save_dataset(df, data_path)
        print(f"Dataset exportado exitosamente en: {saved}")
        return 0

    # Caso 2: Inferencia por lotes (Batch CSV)
    if args.predict_csv:
        in_csv = Path(args.predict_csv)
        out_csv = Path(args.output_csv) if args.output_csv else in_csv.parent / f"predictions_{in_csv.name}"
        print(f"\nEjecutando inferencia por lotes desde {in_csv}...")
        df_out = predict_batch(model_path, in_csv, out_csv)
        print(f"Predicciones generadas para {len(df_out)} registros y guardadas en: {out_csv}")
        return 0

    # Caso 3: Inferencia de muestra individual
    if args.predict or any(x is not None for x in [args.temp, args.ph, args.brix, args.do, args.rpm, args.time, args.inoculum, args.aeration]):
        sample = {
            "temperature_c": args.temp if args.temp is not None else 31.0,
            "ph_level": args.ph if args.ph is not None else 5.8,
            "initial_brix": args.brix if args.brix is not None else 18.0,
            "dissolved_oxygen_pct": args.do if args.do is not None else 50.0,
            "agitation_rpm": args.rpm if args.rpm is not None else 400.0,
            "fermentation_time_h": args.time if args.time is not None else 60.0,
            "inoculum_od600": args.inoculum if args.inoculum is not None else 2.0,
            "aeration_vvm": args.aeration if args.aeration is not None else 1.0,
        }
        res = predict_single(model_path, sample)
        print("\nResultado de Inferencia Individual:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    # Caso 4: Ejecución predeterminada o explícita de pipeline
    return run_full_pipeline(
        n_samples=args.samples,
        data_path=data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
        output_dir=output_dir,
        tune=args.tune,
    )


if __name__ == "__main__":
    sys.exit(main())
