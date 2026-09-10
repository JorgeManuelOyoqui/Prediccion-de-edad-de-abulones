# ============================================================
# Regresión Lineal Sin Frameworks
# Jorge Manuel Oyoqui Aguilera | A01711783
# ============================================================

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42

# ============================================================
# 2. CARGA DEL DATASET
# ============================================================

def load_dataset(data_path: Path) -> pd.DataFrame:
    columns = [
        "Sex", "Length", "Diameter", "Height",
        "Whole", "Shucked", "Viscera", "Shell", "Rings"
    ]
    return pd.read_csv(data_path, names=columns)

# ============================================================
# 3. LIMPIEZA DE DATOS
# ============================================================
 
def detect_outliers_iqr(df: pd.DataFrame, exclude_cols=("Rings",)) -> set:
    # Detect los posibles outliers usando el rango intercuartílico IQR
    # (Q1 - 1.5*IQR, Q3 + 1.5*IQR) en las columnas numéricas
    numerical_cols = (
        df.select_dtypes(include=["float64", "int64"])
        .columns.drop(list(exclude_cols), errors="ignore")
    )
 
    outlier_indices = set()
    for col in numerical_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.update(col_outliers)
 
    return outlier_indices
 
 
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Elimino los registros con Height = 0, ya que son valores inconsistentes
    df = df[df["Height"] != 0]
 
    # Reviso los posibles outliers (rango intercuartílico)
    # Decidí dejar a la mayoría de los posibles outliers, luego
    # de una revisión manual, ya que resultaron ser abulones con valores
    # extremos pero consistentes
    outliers = detect_outliers_iqr(df)
    print(f"Posibles outliers detectados (IQR): {len(outliers)}")
 
    # El único outlier que sí elimino es el del registro 2051, ya que tiene
    # una altura (1.13mm) irregular e inconsistente con el resto de sus
    # características físicas
    df = df.drop(index=2051)
 
    print(f"Número de filas tras la limpieza: {len(df)}")
    return df
 
 
# ============================================================
# 4. PREPARACIÓN DE VARIABLES (X, Y) Y ONE-HOT ENCODING
# ============================================================

# Separo las variables en conjuntos X y Y
def prepare_features(df: pd.DataFrame):
    df_x = df[["Sex", "Length", "Diameter", "Height",
               "Whole", "Shucked", "Viscera", "Shell"]]
    df_y = df[["Rings"]]
 
    # Hago One-hot encoding a Sex, usando Sex_F como categoría de referencia
    df_x_encoded = pd.get_dummies(df_x, columns=["Sex"], drop_first=True)
    df_x_encoded = df_x_encoded.astype(float)
 
    return df_x_encoded, df_y
 
 
# ============================================================
# 5. ANÁLISIS DE CORRELACIÓN
# ============================================================

# Calculo y grafico la matriz de correlación entre las variables
def plot_correlation_heatmap(df_x_encoded: pd.DataFrame, df_y: pd.DataFrame):
    combined_df = pd.concat([df_x_encoded, df_y], axis=1)
    correlation_matrix = combined_df.corr()
 
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix.round(2), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap de la Matriz de Correlación")
    plt.show()
 
 
# ============================================================
# 6. DIVISIÓN EN TRAIN(70%), VALIDATION(15%) Y TEST(15%)
# ============================================================
 
def split_data(df_x_encoded: pd.DataFrame, df_y: pd.DataFrame):

    # Defino la semilla 42 para tener siempre la misma división de datos
    random.seed(RANDOM_SEED)

    # Creo mi lista de registros que van del 0 al 4173
    indices = list(range(len(df_x_encoded)))
    # Mezclo aleatoriamente los índices
    random.shuffle(indices)

    # Defino las proporciones de cada conjunto
    train_ratio, test_ratio = 0.70, 0.15

    # Primero separo el 15% que será utilizado como conjunto de prueba
    test_size = int(len(indices) * test_ratio)
    test_indices = indices[:test_size]

    # El 85% restante se utilizará para Train y Validation
    remaining_indices = indices[test_size:]
    # De los registros restantes, el 70% del total corresponde a Train
    train_size = int(len(indices) * train_ratio)
    train_indices = remaining_indices[:train_size]
    # Los registros restantes corresponden a Validation
    validation_indices = remaining_indices[train_size:]

    # Utilizo los índices para conseguir los conjunto de X y Y
    # EL conjunto X contiene las variables predictoras
    x_train = df_x_encoded.iloc[train_indices]
    x_validation = df_x_encoded.iloc[validation_indices]
    x_test = df_x_encoded.iloc[test_indices]

    # EL conjunto Y contiene la variable objetivo (Rings)
    y_train = df_y.iloc[train_indices].to_numpy().flatten()
    y_validation = df_y.iloc[validation_indices].to_numpy().flatten()
    y_test = df_y.iloc[test_indices].to_numpy().flatten()

    # Muestro las dimensiones de cada conjunto
    print(f"x_train: {x_train.shape} | x_validation: {x_validation.shape} | x_test: {x_test.shape}")
 
    return x_train, x_validation, x_test, y_train, y_validation, y_test
 
 
# ============================================================
# 7. ESTANDARIZACIÓN (usando sólo estadísticas de train)
# ============================================================

# Calculo la media y desviación estándar de x_train para usarlos
# para estandarizar todos los conjuntos (train, validation y test)
def standardize(x_train, x_validation, x_test):
    means = x_train.mean()
    stds = x_train.std()
 
    x_train_scaled = (x_train - means) / stds
    x_validation_scaled = (x_validation - means) / stds
    x_test_scaled = (x_test - means) / stds
 
    return x_train_scaled, x_validation_scaled, x_test_scaled
 
 
# ============================================================
# 8. FUNCIONES DEL MODELO: PREDICCIÓN, MSE, R² Y GRADIENTE
# ============================================================
 
def predict(X, weights, bias):
    return np.dot(X, weights) + bias
 
# Creo la función de costo MSE
def mse(y_real, y_pred):
    n = len(y_real)
    error = y_real - y_pred
    return np.sum(error ** 2) / n
 
# Creo la función para calcular R^2
def r2(y_real, y_pred):
    media = np.mean(y_real)
    suma_errores = np.sum((y_real - y_pred) ** 2)
    suma_total = np.sum((y_real - media) ** 2)
    return 1 - (suma_errores / suma_total)
 
# Creo la función para calcular el gradiente
def calculate_gradient(X, y_real, y_pred):
    # Obtengo los datos necesarios para calcular el gradiente
    n = len(y_real)
    error = y_real - y_pred
    # Calculo el gradiente para los pesos
    # La x transpuesta es para que la multiplicación
    # ∑ x_ij​ (y_i​−y^_​i​) se pueda hacer
    gradient_weights = (-2 / n) * np.dot(X.T, error)
    # Calculo el gradiente para el bias
    gradient_bias = (-2 / n) * np.sum(error)
    return gradient_weights, gradient_bias
 
 
# ============================================================
# 9. ENTRENAMIENTO CON GRADIENTE DESCENDENTE
# ============================================================

# Entreno la regresión lineal múltiple con gradiente descendente,
# revisando el desempeño en train y validation en cada época
def train_linear_regression(
    x_train, y_train, x_validation, y_validation,
    learning_rate, epochs, track_r2=False, verbose=True
):
    # Defino mis pesos y el bias con pesos de 
    weights = np.zeros(x_train.shape[1])
    bias = 0.0

    # Defino las listas que almacenarán el valor del MSE y R^2 a lo largo 
    # de las iteraciones, lo cual usaré posteriormente para la generación 
    # de una gráfica de aprendizaje
    history = {"train_mse": [], "validation_mse": []}
    if track_r2:
        history["train_r2"] = []
        history["validation_r2"] = []
 
    for epoch in range(epochs):
        # Realizo la predicción sobre el conjunto de entrenamiento
        y_train_pred = predict(x_train, weights, bias)
        # Calculo el costo del conjunto de entrenamiento
        train_cost = mse(y_train, y_train_pred)

        # Realizo la predicción sobre el conjunto de validación
        y_validation_pred = predict(x_validation, weights, bias)
        # Calculo el costo del conjunto de validación
        validation_cost = mse(y_validation, y_validation_pred)

        # Guardo los costos de ambos conjuntos
        history["train_mse"].append(train_cost)
        history["validation_mse"].append(validation_cost)

        # Calculo el R^2 para ambos conjuntos si se requiere
        if track_r2:
            history["train_r2"].append(r2(y_train, y_train_pred))
            history["validation_r2"].append(r2(y_validation, y_validation_pred))

        # Calculo el gradiente utilizando ÚNICAMENTE el conjunto de entrenamiento
        gradient_weights, gradient_bias = calculate_gradient(x_train, y_train, y_train_pred)
        # Actualizo los pesos y el bias
        weights = weights - learning_rate * gradient_weights
        bias = bias - learning_rate * gradient_bias

        # Imprimo los valores de MSE cada 100 epochs, y también en la última epoch
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: Train MSE = {train_cost:.4f}, Validation MSE = {validation_cost:.4f}")
 
    return weights, bias, history
 
 
def evaluate_model(weights, bias, x_train, y_train, x_validation, y_validation, x_test, y_test):
    # Realizo las predicciones finales para los tres conjuntos
    y_train_pred = predict(x_train, weights, bias)
    y_validation_pred = predict(x_validation, weights, bias)
    y_test_pred = predict(x_test, weights, bias)
 
    metrics = {
        # Calculo el MSE para cada conjunto
        "train_mse": mse(y_train, y_train_pred),
        "validation_mse": mse(y_validation, y_validation_pred),
        "test_mse": mse(y_test, y_test_pred),
        # Calculo el R² para cada conjunto
        "train_r2": r2(y_train, y_train_pred),
        "validation_r2": r2(y_validation, y_validation_pred),
        "test_r2": r2(y_test, y_test_pred),
    }
    # Calculo el RMSE para cada conjunto
    metrics["train_rmse"] = np.sqrt(metrics["train_mse"])
    metrics["validation_rmse"] = np.sqrt(metrics["validation_mse"])
    metrics["test_rmse"] = np.sqrt(metrics["test_mse"])
 
    return metrics, y_test_pred
 
# Muestro los resultados
def print_metrics(label, metrics):
    print(f"\n--- {label} ---")
    print(f"MSE  -> train: {metrics['train_mse']:.4f} | validation: {metrics['validation_mse']:.4f} | test: {metrics['test_mse']:.4f}")
    print(f"RMSE -> train: {metrics['train_rmse']:.4f} | validation: {metrics['validation_rmse']:.4f} | test: {metrics['test_rmse']:.4f}")
    print(f"R²   -> train: {metrics['train_r2']:.4f} | validation: {metrics['validation_r2']:.4f} | test: {metrics['test_r2']:.4f}")
 
 
# ============================================================
# 10. GRÁFICAS DE ENTRENAMIENTO Y RESULTADOS
# ============================================================
 
def plot_training_history(history, title_suffix=""):
    plt.plot(history["train_mse"], label="Train")
    plt.plot(history["validation_mse"], label="Validation")
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.title(f"Descenso del MSE durante el entrenamiento {title_suffix}".strip())
    plt.legend()
    plt.show()
 
    if "train_r2" in history:
        plt.plot(history["train_r2"], label="Train")
        plt.plot(history["validation_r2"], label="Validation")
        plt.xlabel("Iteración")
        plt.ylabel("R²")
        plt.title(f"Aumento del R² durante el entrenamiento {title_suffix}".strip())
        plt.legend()
        plt.show()
 
 
def plot_predictions_vs_real(y_test, y_test_pred, test_mse):
    plt.scatter(y_test, y_test_pred, alpha=0.6, label="Predicciones del Modelo")
 
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predicción Ideal")
 
    plt.xlabel("Valores Reales de Rings")
    plt.ylabel("Valores Predichos de Rings")
    plt.title(f"Valores Reales vs. Valores Predichos de Rings en el Conjunto de Prueba (MSE: {test_mse:.2f})")
    plt.grid(True)
    plt.legend()
    plt.show()
 
 
# ============================================================
# 11. COMPARACIÓN CONTRA UN BASELINE (promedio de Rings)
# ============================================================
 
def compare_with_baseline(y_train, y_validation, y_test, metrics):
    baseline_prediction = np.mean(y_train)
 
    baseline_train_mse = mse(y_train, np.full(len(y_train), baseline_prediction))
    baseline_validation_mse = mse(y_validation, np.full(len(y_validation), baseline_prediction))
    baseline_test_mse = mse(y_test, np.full(len(y_test), baseline_prediction))
 
    comparison_df = pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Validación", "Prueba"],
        "MSE del Modelo": [metrics["train_mse"], metrics["validation_mse"], metrics["test_mse"]],
        "MSE del Baseline": [baseline_train_mse, baseline_validation_mse, baseline_test_mse],
    })
    print("\nComparación del modelo contra el baseline (promedio de Rings):")
    print(comparison_df.round(4))
 
    improvement = (baseline_test_mse - metrics["test_mse"]) / baseline_test_mse * 100
    print(f"Reducción del MSE respecto al baseline (test): {improvement:.3f}%")
 
 
# ============================================================
# 12. TUNING DE HIPERPARÁMETROS (learning rate)
# ============================================================
 
def tune_learning_rate(x_train, y_train, x_validation, y_validation, learning_rates, epochs):
    results = []
    histories = {}
 
    for lr in learning_rates:
        weights, bias, history = train_linear_regression(
            x_train, y_train, x_validation, y_validation,
            learning_rate=lr, epochs=epochs, track_r2=False, verbose=False
        )
        histories[lr] = history
        results.append({
            "Learning Rate": lr,
            "Train MSE": history["train_mse"][-1],
            "Validation MSE": history["validation_mse"][-1],
        })
 
    results_df = pd.DataFrame(results)
    results_df["Gap"] = results_df["Validation MSE"] - results_df["Train MSE"]
    print("\nResultado del tuning de learning rate:")
    print(results_df)
 
    best_lr = results_df.loc[results_df["Validation MSE"].idxmin(), "Learning Rate"]
    print(f"Mejor learning rate (según Validation MSE): {best_lr}")
 
    return best_lr, results_df, histories
 
 
def plot_learning_rate_histories(histories, metric="train_mse", title=""):
    plt.figure(figsize=(10, 6))
    for lr, history in histories.items():
        plt.plot(history[metric], label=f"LR: {lr}")
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
 
 
# ============================================================
# 13. PROGRAMA PRINCIPAL
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
 
    # Fase de dividir Train/Validation/Test y estandarización
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(df_x_encoded, df_y)
    x_train_scaled, x_validation_scaled, x_test_scaled = standardize(x_train, x_validation, x_test)
 
    # Fase de generar el modelo original con learning rate = 0.01 y 1000 épocas
    print("\n=== Entrenamiento del modelo original (learning rate = 0.01, 1000 épocas) ===")
    weights, bias, history = train_linear_regression(
        x_train_scaled, y_train, x_validation_scaled, y_validation,
        learning_rate=0.01, epochs=1000, track_r2=True
    )
    plot_training_history(history)
 
    original_metrics, y_test_pred = evaluate_model(
        weights, bias,
        x_train_scaled, y_train, x_validation_scaled, y_validation, x_test_scaled, y_test
    )
    print_metrics("Modelo original (LR=0.01, 1000 épocas)", original_metrics)
 
    comparison_table = pd.DataFrame({
        "Valores Reales": y_test[:10],
        "Predicciones": y_test_pred[:10],
    })
    print("\nPrimeras 10 predicciones del conjunto de prueba:")
    print(comparison_table.round(2))
 
    plot_predictions_vs_real(y_test, y_test_pred, original_metrics["test_mse"])
 
    # Comparación del modelo orginal contra baseline 
    compare_with_baseline(y_train, y_validation, y_test, original_metrics)
 
    # Prueba de underfitting/overfitting donde aumento el número de épocas a 2000
    print("\n=== Prueba de diagnóstico: 2000 épocas (learning rate = 0.01) ===")
    _, _, history_2000 = train_linear_regression(
        x_train_scaled, y_train, x_validation_scaled, y_validation,
        learning_rate=0.01, epochs=2000, track_r2=False
    )
    plot_training_history(history_2000, title_suffix="(2000 épocas)")
 
    # Tuning de hiperparámetros donde exploro distintos de learning rates
    print("\n=== Tuning de learning rate (2000 épocas fijas) ===")
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs_tuning = 2000
    best_lr, results_df, lr_histories = tune_learning_rate(
        x_train_scaled, y_train, x_validation_scaled, y_validation,
        learning_rates, epochs_tuning
    )
    plot_learning_rate_histories(
        lr_histories, metric="train_mse",
        title="Descenso del MSE de Entrenamiento para diferentes Learning Rates"
    )
    plot_learning_rate_histories(
        lr_histories, metric="validation_mse",
        title="Descenso del MSE de Validación para diferentes Learning Rates"
    )
 
    # Comparación finale entre el modelo original y el modelo ajustado
    print("\n=== Comparación final: modelo original (LR=0.01) vs. modelo ajustado (LR=0.1), 2000 épocas ===")
    final_results = []
    final_histories = {}
    for lr, label in [(0.01, "Original"), (best_lr, "Ajustado")]:
        weights_final, bias_final, history_final = train_linear_regression(
            x_train_scaled, y_train, x_validation_scaled, y_validation,
            learning_rate=lr, epochs=epochs_tuning, track_r2=False, verbose=False
        )
        final_histories[label] = history_final
        metrics_final, _ = evaluate_model(
            weights_final, bias_final,
            x_train_scaled, y_train, x_validation_scaled, y_validation, x_test_scaled, y_test
        )
        metrics_final["Modelo"] = label
        metrics_final["Learning Rate"] = lr
        final_results.append(metrics_final)
 
    final_comparison_df = pd.DataFrame(final_results).set_index("Modelo")
    print("\nComparación final de métricas:")
    print(final_comparison_df.round(4).T)
 
    original_row = final_comparison_df.loc["Original"]
    adjusted_row = final_comparison_df.loc["Ajustado"]
 
    mse_improvement = (original_row["test_mse"] - adjusted_row["test_mse"]) / original_row["test_mse"] * 100
    rmse_improvement = (original_row["test_rmse"] - adjusted_row["test_rmse"]) / original_row["test_rmse"] * 100
    r2_improvement = (adjusted_row["test_r2"] - original_row["test_r2"]) / original_row["test_r2"] * 100
 
    print(f"\nMejora del Test MSE con el modelo ajustado: {mse_improvement:.2f}%")
    print(f"Mejora del Test RMSE con el modelo ajustado: {rmse_improvement:.2f}%")
    print(f"Mejora del Test R² con el modelo ajustado: {r2_improvement:.2f}%")
 
    plt.figure(figsize=(12, 7))
    plt.plot(final_histories["Original"]["train_mse"], label="Original (LR=0.01) - Train MSE")
    plt.plot(final_histories["Original"]["validation_mse"], label="Original (LR=0.01) - Validation MSE", linestyle="--")
    plt.plot(final_histories["Ajustado"]["train_mse"], label=f"Ajustado (LR={best_lr}) - Train MSE")
    plt.plot(final_histories["Ajustado"]["validation_mse"], label=f"Ajustado (LR={best_lr}) - Validation MSE", linestyle="--")
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.title("Comparación del Descenso del MSE: Modelo Original vs. Modelo Ajustado")
    plt.legend()
    plt.grid(True)
    plt.show()
 
    print("\nModelo final seleccionado: regresión lineal múltiple con gradiente descendente, "
          f"learning rate = {best_lr}, épocas = {epochs_tuning}.")
 
 
if __name__ == "__main__":
    main()
 