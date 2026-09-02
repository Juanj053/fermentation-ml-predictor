"""
Evaluación diagnóstica cuantitativa y generación de visualizaciones bioingenieriles.
Portafolio de Bioingeniería - M1: Predictor de Rendimiento (ML)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Configurar backend headless antes de importar pyplot para ejecución desasistida
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    DEFAULT_METRICS_PATH,
    ALT_METRICS_PATH,
    DEFAULT_PLOT_PATH,
    FIGURES_DIR,
)


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula exhaustivamente todas las métricas de error y correlación para el modelo.
    """
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # MAPE protegido contra divisiones por cero
    denom = np.maximum(y_true, 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    
    max_err = float(max_error(y_true, y_pred))
    
    # Coeficiente de correlación de Pearson
    pearson_r, _ = stats.pearsonr(y_true, y_pred)

    return {
        "r2_score": round(r2, 4),
        "rmse_g_l": round(rmse, 4),
        "mae_g_l": round(mae, 4),
        "mape_percent": round(mape, 2),
        "max_error_g_l": round(max_err, 4),
        "pearson_correlation": round(float(pearson_r), 4),
    }


def save_metrics_report(
    metrics_bundle: Dict[str, Any],
    filepath: Path = DEFAULT_METRICS_PATH,
    also_save_alt: bool = True,
) -> Path:
    """
    Guarda el reporte estructurado de métricas en formato JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=4, ensure_ascii=False)

    if also_save_alt and filepath == DEFAULT_METRICS_PATH:
        ALT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ALT_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics_bundle, f, indent=4, ensure_ascii=False)

    return filepath


def plot_parity(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    """
    Panel 1: Gráfico de Paridad (Actual vs. Predicho) con bandas de tolerancia ±10%.
    """
    min_val = min(np.min(y_true), np.min(y_pred)) * 0.95
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05

    # Scatter coloreado por valor absoluto del residuo
    residuals = np.abs(y_true - y_pred)
    scatter = ax.scatter(
        y_true,
        y_pred,
        c=residuals,
        cmap="viridis_r",
        alpha=0.75,
        edgecolors="none",
        s=35,
        label="Lotes de Prueba",
    )
    plt.colorbar(scatter, ax=ax, label="|Error Absoluto| (g/L)", shrink=0.8)

    # Línea ideal 1:1
    line_pts = np.linspace(min_val, max_val, 100)
    ax.plot(line_pts, line_pts, "r--", linewidth=1.8, label=r"Paridad Ideal ($y=\hat{y}$)")

    # Bandas de tolerancia ±10%
    ax.plot(line_pts, line_pts * 1.10, "gray", linestyle=":", alpha=0.7, label=r"Banda $\pm 10\%$")
    ax.plot(line_pts, line_pts * 0.90, "gray", linestyle=":", alpha=0.7)
    ax.fill_between(line_pts, line_pts * 0.90, line_pts * 1.10, color="gray", alpha=0.08)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(r"Rendimiento Real Experimental $Y_{exp}$ (g/L)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Rendimiento Predicho por Modelo $\hat{Y}$ (g/L)", fontsize=11, fontweight="bold")
    ax.set_title("A. Gráfico de Paridad (Actual vs. Predicho)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Recuadro informativo con métricas
    textstr = (
        f"$R^2 = {metrics['r2_score']:.3f}$\n"
        f"RMSE = {metrics['rmse_g_l']:.2f} g/L\n"
        f"MAE = {metrics['mae_g_l']:.2f} g/L\n"
        f"MAPE = {metrics['mape_percent']:.1f}%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#2b5c8f")
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def plot_residuals(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Panel 2: Diagnóstico de Residuos (Residuals vs Predicho e Histograma con curva normal).
    """
    residuals = y_true - y_pred

    ax.scatter(y_pred, residuals, alpha=0.6, color="#1f77b4", edgecolors="w", s=30)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Residuo Cero")
    
    # Líneas de 2 desviaciones estándar
    std_res = np.std(residuals)
    ax.axhline(2 * std_res, color="orange", linestyle=":", alpha=0.8, label=r"$\pm 2\sigma$")
    ax.axhline(-2 * std_res, color="orange", linestyle=":", alpha=0.8)

    ax.set_xlabel(r"Rendimiento Predicho $\hat{Y}$ (g/L)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Residuos ($Y_{exp} - \hat{Y}$) (g/L)", fontsize=11, fontweight="bold")
    ax.set_title("B. Análisis de Residuos vs. Predicción", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


def plot_feature_importance(
    ax: plt.Axes,
    pipeline: Any,
    feature_names: list,
) -> None:
    """
    Panel 3: Importancia de Variables del Bioproceso (Feature Importance).
    """
    regressor = None
    if hasattr(pipeline, "named_steps") and "regressor" in pipeline.named_steps:
        regressor = pipeline.named_steps["regressor"]
    elif hasattr(pipeline, "feature_importances_"):
        regressor = pipeline

    if regressor is not None and hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif regressor is not None and hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
    else:
        # Importancia uniforme de contingencia
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Nombres amigables
    pretty_names = [FEATURE_METADATA.get(fn, {}).get("name", fn) for fn in feature_names]

    df_imp = pd.DataFrame({"Feature": pretty_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    colors = sns.color_palette("Blues_r", len(df_imp))
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"] * 100.0, color=colors, edgecolor="#1e3d59")
    
    # Anotación del porcentaje
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#2c3e50",
        )

    ax.set_xlim(0, max(df_imp["Importance"] * 100.0) * 1.25)
    ax.set_xlabel("Importancia Relativa (%)", fontsize=11, fontweight="bold")
    ax.set_title("C. Jerarquía de Importancia de Variables", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)


def plot_response_surface(
    ax: plt.Axes,
    pipeline: Any,
    feature_names: list,
) -> None:
    """
    Panel 4: Superficie de Respuesta 2D (Temperatura vs. pH) manteniendo el resto en el óptimo biofísico.
    Demuestra la captura del óptimo biológico no lineal por parte del modelo.
    """
    t_vals = np.linspace(22.0, 40.0, 80)
    ph_vals = np.linspace(4.0, 8.0, 80)
    T_grid, PH_grid = np.meshgrid(t_vals, ph_vals)

    # Crear matriz de evaluación con valores nominales óptimos
    nominal_row = {
        "temperature_c": 31.0,
        "ph_level": 5.8,
        "initial_brix": 18.0,
        "dissolved_oxygen_pct": 50.0,
        "agitation_rpm": 400.0,
        "fermentation_time_h": 60.0,
        "inoculum_od600": 2.0,
        "aeration_vvm": 1.0,
    }

    grid_samples = []
    for t, p in zip(T_grid.ravel(), PH_grid.ravel()):
        row = nominal_row.copy()
        row["temperature_c"] = t
        row["ph_level"] = p
        grid_samples.append([row[fn] for fn in feature_names])

    grid_df = pd.DataFrame(grid_samples, columns=feature_names)
    yield_pred = pipeline.predict(grid_df).reshape(T_grid.shape)

    contour = ax.contourf(T_grid, PH_grid, yield_pred, levels=20, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label("Rendimiento Estimado (g/L)", fontsize=10)

    # Líneas de contorno con etiquetas numéricas
    cs = ax.contour(T_grid, PH_grid, yield_pred, levels=8, colors="white", alpha=0.4, linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    # Marcador del óptimo fisiológico
    max_idx = np.unravel_index(np.argmax(yield_pred), yield_pred.shape)
    opt_t = T_grid[max_idx]
    opt_ph = PH_grid[max_idx]
    opt_yield = yield_pred[max_idx]
    ax.scatter(
        [opt_t],
        [opt_ph],
        color="red",
        marker="*",
        s=180,
        edgecolors="white",
        label=f"Óptimo ({opt_t:.1f}°C, pH {opt_ph:.2f}, {opt_yield:.1f} g/L)",
    )

    ax.set_xlabel("Temperatura (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("pH del Medio", fontsize=11, fontweight="bold")
    ax.set_title("D. Superficie de Respuesta 2D (Temperatura vs. pH)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.3, color="white")


def generate_diagnostic_suite(
    model_bundle: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_composite_path: Path = DEFAULT_PLOT_PATH,
    figures_dir: Optional[Path] = FIGURES_DIR,
) -> Tuple[Path, Dict[str, Path]]:
    """
    Genera la suite completa de figuras diagnósticas (composite 4-panel de 300 DPI y figuras individuales).
    """
    pipeline = model_bundle["pipeline"]
    feature_names = model_bundle.get("feature_names", FEATURE_COLUMNS)
    
    X_te = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=feature_names)
    y_true = np.array(y_test)
    y_pred = pipeline.predict(X_te)
    
    metrics = compute_metrics_dict(y_true, y_pred)

    # 1. Crear figura maestra combinada de 4 paneles
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.25)

    plot_parity(axes[0, 0], y_true, y_pred, metrics)
    plot_residuals(axes[0, 1], y_true, y_pred)
    plot_feature_importance(axes[1, 0], pipeline, feature_names)
    plot_response_surface(axes[1, 1], pipeline, feature_names)

    model_name = model_bundle.get("model_name", "Best Model")
    fig.suptitle(
        f"Diagnóstico de Rendimiento de Fermentación — Modelo: {model_name} ($R^2={metrics['r2_score']:.3f}$)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    output_composite_path = Path(output_composite_path)
    output_composite_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_composite_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # 2. Generar figuras individuales si se especifica directorio
    individual_paths: Dict[str, Path] = {}
    if figures_dir:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # A. Parity Plot
        fig_p, ax_p = plt.subplots(figsize=(8, 7), dpi=300)
        plot_parity(ax_p, y_true, y_pred, metrics)
        p_path = figures_dir / "parity_plot.png"
        fig_p.savefig(p_path, bbox_inches="tight", dpi=300)
        plt.close(fig_p)
        individual_paths["parity_plot"] = p_path

        # B. Residuals Analysis (Left: vs Pred, Right: Histograma)
        fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        plot_residuals(axes_r[0], y_true, y_pred)
        
        # Histograma con ajuste normal
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, color="#2b5c8f", ax=axes_r[1], stat="density", bins=25)
        # Superponer curva gaussiana teórica
        mu, std = np.mean(residuals), np.std(residuals)
        xmin, xmax = axes_r[1].get_xlim()
        x_norm = np.linspace(xmin, xmax, 100)
        p_norm = stats.norm.pdf(x_norm, mu, std)
        axes_r[1].plot(x_norm, p_norm, "r--", linewidth=1.8, label=rf"Gaussiana ($\mu={mu:.2f}, \sigma={std:.2f}$)")
        axes_r[1].set_xlabel("Residuo (g/L)", fontsize=11, fontweight="bold")
        axes_r[1].set_ylabel("Densidad", fontsize=11, fontweight="bold")
        axes_r[1].set_title("Distribución de Residuos", fontsize=12, fontweight="bold")
        axes_r[1].legend(loc="upper right", fontsize=9)
        axes_r[1].grid(True, linestyle="--", alpha=0.5)

        r_path = figures_dir / "residuals_analysis.png"
        fig_r.savefig(r_path, bbox_inches="tight", dpi=300)
        plt.close(fig_r)
        individual_paths["residuals_analysis"] = r_path

        # C. Feature Importance
        fig_f, ax_f = plt.subplots(figsize=(8, 6), dpi=300)
        plot_feature_importance(ax_f, pipeline, feature_names)
        f_path = figures_dir / "feature_importance.png"
        fig_f.savefig(f_path, bbox_inches="tight", dpi=300)
        plt.close(fig_f)
        individual_paths["feature_importance"] = f_path

        # D. Response Surface 2D
        fig_s, ax_s = plt.subplots(figsize=(9, 7), dpi=300)
        plot_response_surface(ax_s, pipeline, feature_names)
        s_path = figures_dir / "response_surface_2d.png"
        fig_s.savefig(s_path, bbox_inches="tight", dpi=300)
        plt.close(fig_s)
        individual_paths["response_surface_2d"] = s_path

    return output_composite_path, individual_paths
