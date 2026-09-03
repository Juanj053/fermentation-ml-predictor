# 🌿 Predictor de Rendimiento de Fermentación (Machine Learning)
**Módulo 01 — Portafolio de Bioingeniería y Optimización de Bioprocesos**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production-Grade](https://img.shields.io/badge/Status-Production--Grade-brightgreen.svg)]()

---

## 1. Resumen Ejecutivo & Contexto Bioingenieril

En la industria biotecnológica y de fermentaciones de precisión (producción de bioinsumos agrícolas, metabolitos secundarios, enzimas y proteínas recombinantes), el **rendimiento final del producto ($Y$ en $\text{g/L}$)** está gobernado por interacciones cinéticas altamente no lineales y acopladas entre variables fisicoquímicas e hidrodinámicas del biorreactor.

Las aproximaciones estadísticas lineales clásicas (como regresión lineal múltiple o modelos polinomiales de bajo orden) fracasan sistemáticamente ($R^2 < 0.25$) debido a fenómenos biológicos críticos como:
1. **Desnaturalización térmica irreversible** a temperaturas supra-óptimas.
2. **Inactivación enzimática** por desviación del estado de ionización del sitio activo (curvas de Dixon-Webb).
3. **Inhibición por exceso de sustrato y estrés osmótico** (cinética de Haldane-Andrews).
4. **Limitaciones en la tasa de transferencia de oxígeno ($k_L a$)** combinadas con **daño celular por esfuerzo cortante (shear stress)** a altas velocidades de agitación.

Este proyecto implementa un pipeline integral de Machine Learning en Python que:
- Modela un **Proceso Generador de Datos (DGP)** fundamentado en ecuaciones cinéticas biofísicas reales.
- Realiza un **benchmark riguroso** entre modelos lineales regularizados (`Ridge`) y ensambles no lineales (`Random Forest`, `Gradient Boosting`) mediante **Validación Cruzada de 5 pliegues (5-Fold CV)**.
- Selecciona y serializa el mejor estimador predictivo ($R^2 > 0.90$, $\text{RMSE} < 6.0\text{ g/L}$).
- Genera un **panel de diagnóstico gráfico con 4 visualizaciones** en resolución de publicación (300 DPI) usando renderizado headless (`Agg`).
- Provee un **motor de inferencia CLI** para predicciones individuales (con diagnóstico de régimen operativo) y procesamiento masivo por lotes (Batch CSV).

---

## 2. Fundamentación Matemática del Modelo Cinético (DGP)

El rendimiento final de fermentación $Y (\text{g/L})$ se modela mediante la acción multiplicativa de funciones de respuesta biológica normalizadas $f_i \in [0, 1]$ escaladas por el rendimiento teórico máximo $Y_{\text{max}} = 88.0\text{ g/L}$:

$$Y = Y_{\text{max}} \cdot f_T(T) \cdot f_{\text{pH}}(\text{pH}) \cdot f_S(\text{Brix}) \cdot f_{O_2}(\text{DO}, \text{RPM}, \text{vvm}) \cdot f_t(t) \cdot f_{\text{inoc}}(OD) + \epsilon$$

### 2.1 Variables Operativas y Ventanas Fisiológicas

| Variable | Notación | Rango Exploratorio | Ventana Óptima | Justificación Fisiológica |
| :--- | :--- | :--- | :--- | :--- |
| **Temperatura** | `temperature_c` | 22.0 – 40.0 °C | 30.0 – 32.0 °C | Cinética de Arrhenius y desnaturalización térmica ($T_{\text{opt}} = 31.0^\circ\text{C}$). |
| **pH del Caldo** | `ph_level` | 4.0 – 8.0 | 5.5 – 6.2 | Ionización de aminoácidos catalíticos y gradiente de protones ($\text{pH}_{\text{opt}} = 5.8$). |
| **Azúcares Iniciales** | `initial_brix` | 10.0 – 26.0 °Brix | 16.0 – 20.0 °Brix | Fuente de carbono. Saturación Monod e inhibición Haldane a $>22^\circ\text{Bx}$. |
| **Oxígeno Disuelto** | `dissolved_oxygen_pct` | 10.0 – 90.0 % | 30.0 – 60.0 % | Aceptor terminal de electrones en fosforilación oxidativa. |
| **Agitación** | `agitation_rpm` | 150 – 600 RPM | 350 – 450 RPM | Coeficiente $k_L a$ de transferencia de masa; daño por cizalla $>550\text{ RPM}$. |
| **Tiempo de Proceso** | `fermentation_time_h` | 18.0 – 96.0 h | 48.0 – 72.0 h | Acumulación sigmoidal logística de producto. |
| **Inóculo Inicial** | `inoculum_od600` | 0.5 – 3.0 $\text{OD}_{600}$ | 1.5 – 2.2 $\text{OD}_{600}$ | Modulación de la duración de la fase lag. |
| **Tasa de Aireación** | `aeration_vvm` | 0.2 – 2.0 vvm | 0.8 – 1.2 vvm | Flujo volumétrico de aire por volumen de líquido por minuto. |

---

### 2.2 Ecuaciones Cinéticas Parciales

#### A. Respuesta Térmica Cardinal ($f_T$)
Modela la activación térmica de Arrhenius y la desactivación enzimática abrupta por encima de 38.0°C:
$$f_T(T) = \exp\left( - \frac{(T - 31.0)^2}{2 \cdot (3.5)^2} \right) \cdot \begin{cases} 1.0 & \text{si } T \le 38.0^\circ\text{C} \\ \exp\left(-0.8 \cdot (T - 38.0)\right) & \text{si } T > 38.0^\circ\text{C} \end{cases}$$

#### B. Ionización Catalítica de Dixon-Webb ($f_{\text{pH}}$)
$$f_{\text{pH}}(\text{pH}) = \exp\left( - \frac{(\text{pH} - 5.8)^2}{2 \cdot (0.75)^2} \right)$$

#### C. Cinética de Sustrato de Haldane-Andrews ($f_S$)
$$f_S(\text{Brix}) = 2.2 \cdot \frac{\text{Brix}}{K_s + \text{Brix} + \frac{\text{Brix}^2}{K_i}}, \quad K_s = 3.0^\circ\text{Bx}, \; K_i = 35.0^\circ\text{Bx}$$

#### D. Transferencia de Oxígeno ($k_L a$) y Esfuerzo Cortante ($f_{O_2}$)
$$\kappa_{O_2} = \left(\frac{\text{RPM}}{400}\right)^{0.8} \cdot \left(\frac{\text{DO}\%}{50}\right)^{0.5} \cdot \left(\frac{\text{vvm}}{1.0}\right)^{0.3}$$
$$f_{O_2} = \min\left(1.0, \frac{\kappa_{O_2}}{1.1}\right) \cdot \left[1.0 - \max\left(0, \frac{\text{RPM} - 550}{300}\right)\right]$$

#### E. Acumulación Temporal Logística ($f_t$)
$$f_t(t) = \frac{1}{1 + \exp\left( -0.085 \cdot (t - 38.0) \right)}$$

#### F. Ruido Estocástico y Fallas Operativas
- **Ruido Gaussiano**: $\epsilon \sim \mathcal{N}(0, 1.8^2)\text{ g/L}$.
- **Anomalías de Lote**: Tasa estocástica del 2.0% donde el rendimiento colapsa ($Y \times 0.25$) por contaminación o fallo electromecánico.

---

## 3. Arquitectura del Sistema & Estructura de Archivos

```
01_predictor_rendimiento_ml/
├── data/
│   └── synthetic_fermentation_data.csv  # 1,500 lotes generados con cinéticas biofísicas
├── models/
│   └── best_yield_predictor.joblib      # Modelo optimizado serializado con metadatos
├── reports/
│   ├── metrics.json                     # Resumen estructurado de métricas y benchmark
│   ├── evaluation_plots.png             # Panel diagnóstico combinado (4 paneles, 300 DPI)
│   └── figures/                         # Figuras individuales en alta resolución
│       ├── parity_plot.png
│       ├── residuals_analysis.png
│       ├── feature_importance.png
│       └── response_surface_2d.png
├── src/
│   ├── __init__.py
│   ├── config.py                        # Parámetros cinéticos, rangos y rutas globales
│   ├── data_generator.py                # Motor generador de datos basado en cinética
│   ├── model_pipeline.py                # Pipelines scikit-learn, benchmark y 5-Fold CV
│   ├── evaluate.py                      # Cálculo de métricas y generador gráfico headless
│   └── predict.py                       # Motor de inferencia individual y batch
├── main.py                              # CLI principal y punto de entrada sin configuración
├── requirements.txt                     # Dependencias fijadas
└── README.md                            # Documentación técnica completa
```

---

## 4. Instalación y Requisitos

### 4.1 Entorno Virtual con `pip`
```bash
cd portafolio_bioingenieria/01_predictor_rendimiento_ml
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Ejecución Directa con `uv`
```bash
uv run --with scikit-learn,numpy,pandas,matplotlib,seaborn,joblib,scipy python3 main.py
```

---

## 5. Guía de Uso y Comandos CLI

El script `main.py` incorpora una interfaz de línea de comandos potente y flexible.

### 5.1 Ejecución Completa de Punta a Punta (Zero-Config)
Ejecuta la generación de datos, partición 80/20, benchmark de 3 algoritmos con 5-Fold CV, selección del ganador, evaluación en conjunto de prueba independiente, exportación de reportes/figuras e inferencia de prueba:
```bash
python3 main.py
```

### 5.2 Generación y Exportación de Dataset Personalizado
```bash
# Generar 3,000 lotes y exportar a una ruta específica
python3 main.py --export-dataset --samples 3000 --data-path data/custom_fermentation_dataset.csv
```

### 5.3 Entrenamiento con Optimización de Hiperparámetros (GridSearchCV)
```bash
python3 main.py --train --tune --samples 2000
```

### 5.4 Inferencia para un Lote Individual
Permite ingresar los parámetros operativos del biorreactor y obtener la predicción con su intervalo de confianza al 95% y diagnóstico de régimen operativo:
```bash
python3 main.py --predict \
  --temp 31.2 \
  --ph 5.85 \
  --brix 18.5 \
  --do 52.0 \
  --rpm 410 \
  --time 60.0 \
  --inoculum 2.0 \
  --aeration 1.1
```
**Salida JSON estructurada:**
```json
{
  "predicted_yield_g_l": 66.31,
  "ci_95_lower_g_l": 55.13,
  "ci_95_upper_g_l": 77.49,
  "model_used": "Gradient_Boosting",
  "operational_regime": "Condición Óptima",
  "is_safe": true,
  "warnings": [],
  "input_features": {
    "temperature_c": 31.2,
    "ph_level": 5.85,
    "initial_brix": 18.5,
    "dissolved_oxygen_pct": 52.0,
    "agitation_rpm": 410.0,
    "fermentation_time_h": 60.0,
    "inoculum_od600": 2.0,
    "aeration_vvm": 1.1
  }
}
```

### 5.5 Inferencia Masiva por Lotes (Batch CSV)
```bash
python3 main.py --predict-csv data/synthetic_fermentation_data.csv --output-csv data/batch_predictions.csv
```

---

## 6. Resultados Cuantitativos del Benchmark

El benchmark evalúa $N = 1,500$ lotes ($1,200$ entrenamiento, $300$ prueba independiente) sobre las 8 características bioingenieriles:

| Modelo Evaluado | Paradigma Algorítmico | CV $R^2$ (5-Fold Mean ± Std) | Test $R^2$ | Test RMSE (g/L) | Test MAE (g/L) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Lineal Regularizado L2 | $0.202 \pm 0.035$ | $0.222$ | $16.38\text{ g/L}$ | $12.75\text{ g/L}$ |
| **Random Forest** | Ensamble Bagging (150 árboles) | $0.886 \pm 0.009$ | $0.868$ | $6.74\text{ g/L}$ | $4.52\text{ g/L}$ |
| **Gradient Boosting** 🏆 | Ensamble Boosting MSE | $\mathbf{0.913 \pm 0.009}$ | $\mathbf{0.906}$ | $\mathbf{5.71\text{ g/L}}$ | $\mathbf{3.84\text{ g/L}}$ |

### Conclusiones del Benchmark:
1. **Fallo del modelo lineal**: El modelo Ridge solo explica el $22.2\%$ de la variabilidad del bioproceso, confirmando que las no linealidades enzimáticas y de transporte no pueden modelarse con hiperplanos simples.
2. **Superioridad del Gradient Boosting**: Alcanza un coeficiente de determinación **$R^2 = 0.9056$** en el conjunto de prueba independiente y un error medio absoluto de apenas **$3.84\text{ g/L}$**, con un coeficiente de correlación de Pearson $r = 0.9518$.

---

## 7. Interpretación de los Gráficos Diagnósticos

El pipeline genera automáticamente el panel maestro de 4 cuadrantes en `reports/evaluation_plots.png`:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  A. Gráfico de Paridad (1:1)         │  B. Análisis de Residuos             │
│  - Correlación y bandas ±10%         │  - Homocedasticidad y normalidad     │
├──────────────────────────────────────┼──────────────────────────────────────┤
│  C. Importancia de Variables (%)     │  D. Superficie de Respuesta 2D       │
│  - Jerarquía de drivers biofísicos   │  - Contorno Temperatura vs. pH       │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

1. **Gráfico de Paridad (Actual vs. Predicho)**:
   - Los puntos de prueba se alinean estrechamente sobre la diagonal $y = x$.
   - La gran mayoría de los lotes se sitúan dentro de la banda de tolerancia de $\pm 10\%$.
2. **Análisis de Residuos**:
   - Dispersión aleatoria en torno al residuo cero sin curvaturas sistemáticas, demostrando homocedasticidad.
   - El histograma de residuos sigue una distribución normal ajustada centrada en cero.
3. **Jerarquía de Importancia de Variables**:
   - Identifica a la **Temperatura**, **pH** y **Tiempo de Fermentación** como las variables dominantes en la determinación del rendimiento final, seguidas por la **concentración de sustrato (°Brix)** y las condiciones de **oxigenación/agitación**.
4. **Superficie de Respuesta 2D (Temperatura vs. pH)**:
   - Reconstruye visualmente la campana de rendimiento biológico, alcanzando el pico de máxima productividad ($>66\text{ g/L}$) en la vecindad de $T = 31.0^\circ\text{C}$ y $\text{pH} = 5.80$.

---

## 8. Licencia y Créditos

Desarrollado como parte del **Portafolio Abierto de Bioingeniería y Optimización de Bioprocesos Industriales**.  
Licencia MIT. Libre para uso académico, industrial y de investigación.
