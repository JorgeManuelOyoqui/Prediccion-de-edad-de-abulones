# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

df_y = df["Rings"]

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

# train_test_split hace automáticamente la división aleatoria de los datos.
# test_size=0.2 significa que el 20% de los datos serán utilizados
# para pruebas y el otro 80% restante serán para entrenamiento.
# random_state=42 permite obtener siempre la misma división.

x_train, x_test, y_train, y_test = train_test_split(
    df_x_encoded,
    df_y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# 8. ESTANDARIZACIÓN 
# ============================================================

# Creo el objeto encargado de estandarizar las variables.
scaler = StandardScaler()

# Ajusto el scaler únicamente con los datos de entrenamiento y transformo dichos datos.
x_train_scaled = scaler.fit_transform(x_train)

# Utilizo los mismos parámetros obtenidos del conjunto de entrenamiento 
# para transformar el conjunto de prueba.
x_test_scaled = scaler.transform(x_test)

# ============================================================
# 9. CREACIÓN DEL MODELO
# ============================================================

# Creo una instancia para el modelo de regesión lineal y así sea más intuitivo de usar.
model = LinearRegression()

# ============================================================
# 10. ENTRENAMIENTO DEL MODELO
# ============================================================

# Aquí el framework calcula automáticamente los parámetros
# del modelo a usando de referencia los datos del entrenamiento.
model.fit(x_train_scaled, y_train)

# ============================================================
# 11. PREDICCIONES
# ============================================================

# Genero las predicciones para los conjuntos de entrenamiento y de prueba
y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)

# ============================================================
# 12. EVALUACIÓN DEL MODELO
# ============================================================

# Calculo el MSE del conjunto de entrenamiento.
train_mse = mean_squared_error(y_train, y_train_pred)

# Calculo el MSE del conjunto de prueba.
test_mse = mean_squared_error(y_test, y_test_pred)

# Calculo el RMSE para tener el error pero en las mismas unidades que Rings.
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print(f"MSE de entrenamiento: {train_mse:.4f}")
print(f"MSE de prueba: {test_mse:.4f}")

print(f"RMSE de entrenamiento: {train_rmse:.4f}")
print(f"RMSE de prueba: {test_rmse:.4f}")

# ============================================================
# 13. COMPARACIÓN DE PREDICCIONES
# ============================================================

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

# A diferencia de la versión sin framework, aquí no tengo mse_history 
# porque la LinearRegression encuentra directamente los 
# valores usando mínimos cuadrados.
# Por ende, en ese caso no voy a tener una gráfica de descenso del MSE,
# sino una gráfica que compare los valores reales con los predichos por mi modelo.

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test_pred, alpha=0.6)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

# Línea de predicción perfecta.
plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
plt.xlabel("Valores Reales de Rings")
plt.ylabel("Valores Predichos de Rings")
plt.title(f"Valores Reales vs. Valores Predichos " f"(MSE: {test_mse:.2f})")
plt.grid(True)
plt.show()

