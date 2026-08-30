# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 2. CARGA DEL DATASET
# ============================================================

columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height", 
    "Whole", 
    "Shucked", 
    "Viscera", 
    "Shell", 
    "Rings"
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "abalone" / "abalone.data"

df = pd.read_csv(DATA_PATH, names = columns)

# ============================================================
# 3. LIMPIEZA DE DATOS
# ============================================================

# Elimino los registros con Height = 0
# ya que son valores inconsistentes para dicha variable
df = df[df['Height'] != 0]

# Obtengo las columnas con valores numéricos que usaré para detectar posibles outliers
# Excluyo a Rings por que es la variable objetivo (o a predecir)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Rings', errors='ignore')
#Genero una lista vacía para guardar la cantidad de posibles outliers
outlier_indices = []

# Uso cuartíles donde, si los valores salen del rango de 
# Q1-1.5*IQR a Q3+1.5*IQR, entonces se consideran outliers
for col in numerical_cols: #Para cada una de las columnas con valor numérico:
    Q1 = df[col].quantile(0.25) # Defino el primer cuartil
    Q3 = df[col].quantile(0.75) # Defino el tercer cuartil
    IQR = Q3 - Q1 # Rango intercuartílico

    # Si la variable es menor o mayor al rango intercuartílico,
    #se guarda en una de las dos variables según cual sea su caso
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifico a los outliers de la columna actual
    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    outlier_indices.extend(col_outliers)

# Guardo los índices únicos de las filas que tienen al menos un posible outlier
# para revisar manualmente los datos y ver cuáles voy a excluir.
unique_outlier_indices = list(set(outlier_indices))

# Esto lo explico mejor en el reporte, pero elimino sólo al registro 2051 manualmente
# por que, luego de una revisión manual de los posibles outliers, este fue el único
# registro con un valor inconsistente en Height comparado con el resto de sus características.
# El resto de los posibles outliers se mantuvieron debido a la consistencia de sus datos. 
df = df.drop(index=2051)

# ============================================================
# 4. PREPARACIÓN DE LAS VARIABLES X y Y
# ============================================================

df_x = df[[
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole", 
    "Shucked", 
    "Viscera", 
    "Shell"
]]

df_y = df[["Rings"]]

# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

# Realizo one-hot encoding en la columna Sex de df_x.
# Esto transforma a la columa Sex en dos columnas binarias:
# Sex_I y Sex_M, usando a Sex_F como categoría de referencia
df_x_encoded = pd.get_dummies(df_x, columns=['Sex'], drop_first=True)

# Convierto sus variables en tipo numérico
df_x_encoded = df_x_encoded.astype(float)

# ============================================================
# 6. ANÁLISIS DE CORRELACIÓN
# ============================================================

# Combino las características codificadas con la columna 'Rings'
combined_df = pd.concat([df_x_encoded, df_y], axis=1)

# Calculo la matriz de correlación
correlation_matrix = combined_df.corr()

# Genero una heat map para mostrar las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix.round(2), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de la Matriz de Correlación')
plt.show()

# ============================================================
# 7. DIVISIÓN DEL DATASET EN TRAIN Y TEST
# ============================================================

# Defino la semilla 42 para tener siempre la misma división de datos
random.seed(42)

# Creo mi lista de registros que van del 0 al 4173
indices = list(range(len(df_x_encoded)))
# Mezclo los índices
random.shuffle(indices)

# Defino la proporción de entrenamiento
train_ratio = 0.8
# Calculo el número de registros que serán de entrenamiento (3339 registros en total)
train_size = int(len(indices) * train_ratio)

# Defino los índices de entrenamiento y de pruebas usando slicing
train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_train = df_x_encoded.iloc[train_indices]
x_test = df_x_encoded.iloc[test_indices]

y_train = df_y.iloc[train_indices].to_numpy().flatten()
y_test = df_y.iloc[test_indices].to_numpy().flatten()

# ============================================================
# 8. ESTANDARIZACIÓN 
# ============================================================

# Calculo la media y desviación estándar de x_train
means = x_train.mean()
stds = x_train.std()

# Estandarizo los conjuntos de entrenamiento y prueba de X
x_train_scaled = (x_train - means) / stds
x_test_scaled = (x_test - means) / stds

# ============================================================
# 9. FUNCIÓN DE PREDICCIÓN
# ============================================================

# Calcula la predicciones del modelo
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

# ============================================================
# 10. FUNCIÓN DE COSTO
# ============================================================

# Calcula el MSE
def mse(y_real, y_pred):
    n = len(y_real)
    error = y_real - y_pred
    error_cuadrado = error ** 2
    return np.sum(error_cuadrado) / n

# ============================================================
# 11. FUNCIÓN DEL GRADIENTE DESCENDENTE
# ============================================================

# Calcula el los gradientes de los pesos y del bias
def calculate_gradient(X, y_real, y_pred):
    n = len(y_real)
    error = y_real - y_pred

    gradient_weights = (-2 / n) * np.dot(X.T, error)
    gradient_bias = (-2 / n) * np.sum(error)

    return gradient_weights, gradient_bias

# ============================================================
# 12. INICIALIZACIÓN DE HIPERPARÁMETROS
# ============================================================

learning_rate = 0.01
epochs = 1000

weights = np.zeros(x_train_scaled.shape[1])
bias = 0.0
mse_history = []

# ============================================================
# 13. ENTRENAMIENTO DEL MODELO
# ============================================================

for epoch in range(epochs):

    # Realizo la predicción
    y_pred = predict(x_train_scaled, weights, bias)

    # Calculo y guard el MSE
    cost = mse(y_train, y_pred)
    mse_history.append(cost)

    # Calculo los gradientes
    gradient_weights, gradient_bias = calculate_gradient(
        x_train_scaled,
        y_train,
        y_pred
    )

    # Actualizo los parámetros
    weights = weights - learning_rate * gradient_weights
    bias = bias - learning_rate * gradient_bias

# Grafico los resultados del entrenamiento del modelo
plt.plot(mse_history)
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Descenso del MSE durante el entrenamiento")
plt.show()

# ============================================================
# 14. EVALUACIÓN DEL MODELO
# ============================================================

y_test_pred = predict(x_test_scaled, weights, bias)

# Comparo el MSE de mi conjunto de pruebas con el
# MSE del conjunto de entrenamiento
train_mse = mse(y_train, predict(x_train_scaled, weights, bias))
test_mse = mse(y_test, y_test_pred)
print(f"MSE de entrenamiento: {train_mse:.4f}")
print(f"MSE de prueba: {test_mse:.4f}")

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print(f"RMSE de entrenamiento: {train_rmse:.4f}")
print(f"RMSE de prueba: {test_rmse:.4f}")

# Comparo el resultado de mis predicciones
# con los valores reales de los registros
first_10_predictions = y_test_pred[:10]
first_10_actual = y_test[:10]
# Genero un nuevo dataframe que sirva para comparar los resultados
comparison_df = pd.DataFrame({
    'Valores Reales': first_10_actual,
    'Predicciones': first_10_predictions
})
print("\nPrimeras 10 predicciones:")
print(comparison_df.round(2))

# Genero el gráfico que compara los valores de Rings predichos por mi modelo con los reales

plt.scatter(y_test, y_test_pred, alpha=0.6)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.xlabel("Valores Reales de Rings")
plt.ylabel("Valores Predichos de Rings")
plt.title(f"Valores Reales vs. Valores Predichos de Rings en el Conjunto de Prueba (MSE: {test_mse:.2f})")
plt.grid(True)
plt.show()
