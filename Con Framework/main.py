# ============================================================
# Regresión Lineal con frameworks (scikit-learn)
# Jorge Manuel Oyoqui Aguilera | A01711783
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42


# ============================================================
# 1. CARGA DEL DATASET
# ============================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    columns = [
        "Sex", "Length", "Diameter", "Height",
        "Whole", "Shucked", "Viscera", "Shell", "Rings"
    ]
    return pd.read_csv(data_path, names=columns)


# ============================================================
# 2. LIMPIEZA DE DATOS
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Elimino los registros con Height = 0, ya que son valores
    # físicamente inconsistentes para esa variable.
    df = df[df["Height"] != 0]

    # El único outlier que sí elimino es el del registro 2051, ya que tiene
    # una altura (1.13mm) irregular e inconsistente con el resto de sus
    # características físicas
    df = df.drop(index=2051)

    print(f"Número de filas tras la limpieza: {len(df)}")
    return df


# ============================================================
# 3. PREPARACIÓN DE VARIABLES (X, Y) Y ONE-HOT ENCODING
# ============================================================

# Separo las variables en conjuntos X y Y
def prepare_features(df: pd.DataFrame):
    df_x = df[["Sex", "Length", "Diameter", "Height",
               "Whole", "Shucked", "Viscera", "Shell"]]
    df_y = df["Rings"]

    # Hago One-hot encoding a Sex, usando Sex_F como categoría de referencia
    df_x_encoded = pd.get_dummies(df_x, columns=["Sex"], drop_first=True)
    df_x_encoded = df_x_encoded.astype(float)

    return df_x_encoded, df_y


# ============================================================
# 4. ANÁLISIS DE CORRELACIÓN
# ============================================================

# Calculo y grafico la matriz de correlación entre las variables
def plot_correlation_heatmap(df_x_encoded: pd.DataFrame, df_y: pd.Series):
    combined_df = pd.concat([df_x_encoded, df_y], axis=1)
    correlation_matrix = combined_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix.round(2), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap de la Matriz de Correlación")
    plt.show()


# ============================================================
# 5. EVALUACIÓN Y VISUALIZACIÓN DE RESULTADOS
# ============================================================

def print_metrics(label, y_real, y_pred):
    # Caclulo las métricas de evaluación MSE, RMSE y R²
    mse_value = mean_squared_error(y_real, y_pred)
    rmse_value = np.sqrt(mse_value)
    r2_value = r2_score(y_real, y_pred)
    print(f"{label} -> MSE: {mse_value:.4f} | RMSE: {rmse_value:.4f} | R²: {r2_value:.4f}")
    return mse_value, rmse_value, r2_value

# Gráfica de los valores reales contra los predichos
def plot_predictions_vs_real(y_test, y_test_pred, test_mse):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6, label="Predicciones del Modelo")

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predicción Ideal")

    plt.xlabel("Valores Reales de Rings")
    plt.ylabel("Valores Predichos de Rings")
    plt.title(f"Valores Reales vs. Valores Predichos de Rings (MSE: {test_mse:.2f})")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 6. PROGRAMA PRINCIPAL
# ============================================================

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "abalone" / "abalone.data"

    # Fase de ETL y limpieza
    df = load_dataset(DATA_PATH)
    df = clean_dataset(df)

    # Fase de preparación de variables y one-hot encoding
    df_x_encoded, df_y = prepare_features(df)
    plot_correlation_heatmap(df_x_encoded, df_y)

    # Fase de dividir Train(70%), Validation(15%) y Test (15%)
    # Primero separo el 15% que será el conjunto de prueba.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        df_x_encoded, df_y, test_size=0.15, random_state=RANDOM_STATE
    )
    # Del 85% restante, separo la parte que le corresponde a validation
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val, y_train_val, test_size=0.15 / 0.85, random_state=RANDOM_STATE
    )
    print(f"x_train: {x_train.shape} | x_validation: {x_validation.shape} | x_test: {x_test.shape}")

    # Fase de estandarización (ajustada únicamente con train)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    # Fase de creación y entrenamiento del modelo
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    # Fase de predicciones
    y_train_pred = model.predict(x_train_scaled)
    y_validation_pred = model.predict(x_validation_scaled)
    y_test_pred = model.predict(x_test_scaled)

    # Evaluación del modelo
    print()
    print_metrics("Entrenamiento", y_train, y_train_pred)
    print_metrics("Validación", y_validation, y_validation_pred)
    print_metrics("Prueba", y_test, y_test_pred)

    # Comparación de predicciones
    comparison_df = pd.DataFrame({
        "Valores Reales": y_test.iloc[:10].to_numpy(),
        "Predicciones": y_test_pred[:10],
    })
    print("\nPrimeras 10 predicciones:")
    print(comparison_df.round(2))

    # Gráfica de los valores reales contra los predichos
    # A diferencia de la versión sin framework, aquí no tengo un historial de
    # MSE por época ya que LinearRegression resuelve la regresión de forma
    # analítica (con mínimos cuadrados), no de forma iterativa.
    test_mse = mean_squared_error(y_test, y_test_pred)
    plot_predictions_vs_real(y_test, y_test_pred, test_mse)


if __name__ == "__main__":
    main()