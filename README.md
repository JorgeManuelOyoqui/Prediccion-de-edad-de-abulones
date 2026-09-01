# Prediccion-de-edad-de-abulones
Para este proyecto implemento técnicas de Machine Learning para crear un modelo de regresión lineal que prediga la edad de un abulón a través de sus características físicas.
Para este proyecto, generé dos versiones diferentes de crear el modelo de predicción:
- **Sin framework**: Esta es la versión principal, en la que manualmente creo el modelo de regresión lineal usando regresión descente para predecir la edad de los abulones.
- **Con framework**: Donde uso la biblioteca Scikit-learn para crear, entrenar y probar al modelo.

Ambas versiones utilizan el mismo dataset y tienen el mismo proceso de limpieza y preparación de los datos, de modo que como sólo cambia la forma en la que se crea y entrena el modelo, puedo comparar mi modelo manual con el modelo que se creó usando Scikit-learn.

## Dataset
El datatset que utilicé fue el de **Abalone**, el cual se puede encontrar en la siguiente liga: https://archive.ics.uci.edu/dataset/1/abalon

El objetivo del dataset es, como ya mencioné, lograr predecir la edad de un abulón a través de sus características física, donde la variable objetivo a predecir va ser la de `Rings`. La razón es por que la cantidad de anillos de los abulones sirven para estimar su edad.

El resto de las variables son:
- `Sex`
- `Length`
- `Diameter`
- `Height`
- `Whole`
- `Shucked`
- `Viscera`
- `Shell`

## Proceso

Antes de entrenar los modelos, primero se hacen las siguientes etapas:
1. Cargar el dataset
2. Limpiar los datos
3. Detectar posibles outliers usando IQR
4. Revisar manualmente los posibles outliers
5. Eliminar los registros inconsistentes de los psoibles outliers
6. Separar el dataset en las variables X y Y
7. Hacer One-Hot Encpding a la varibale `Sex`
8. Dividr los datos en cnjuntos de entrenamiento y prueba
9. Estandarizar las variable (usando sólo los parámetros que calculé del conjunto de entrenamiento)

## Modelo Sin Framework
La carpeta `Sin Framework` es la que contiene el modelo que no usa frameworks, de mood que ese modelo se crea con una regresión lineal múltiple usando gradient descendente.

Lo que debo implementar manualmente es:
- Inicialización de hiperparámetros.
- Función de predicción.
- Función de costo MSE.
- Cálculo ed los gradientes.
- Actualziación de los pesos y de bias.
- Entrenamiento itrativo.
- Generación de las predicciones.
- Evalución del modelo a través del MSE.

Igualmente, y a diferencia de la versión con framework, genero una grafica que muestra como es que disminuye el MSE en cada época.

## Modelo Con Framework
La carpeta `Con Framework` usa la librería Scikit-learn para hacer la regresión lineal.

En esta versión ya no tengo que implementar varias cosas manualmente, de modo que utilizo:
- `train_test_split` para dividir los datos en conjuntos de entrenamiento y prueba.
- `StandardScaler` para estandarizar las variables.
- `LinearRegression` para entrenar el modelo.
- `mean_squared_error` para evaluar las predicciones.
Y aquí, la regresión linal la hace Scikit-learn sin que yo la tenga que hacer manualmente.

## Requisitos
Necesitas lo siguiene para poder correr este proyecto:
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Instalación
Recomiendo usar un entorno virtual, ya que eso es lo que hice yo con:
```bash
python -m venv env
```

Para activar el entorno virtual, corre: 
```bash
.\env\Scripts\activate
```

Y corre lo siguiente en la terminal para instalar las dependencias que este proyecto va a usar:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Ejectura código
Para ejecutar el modelo sin framework, corre en la terminal:
```bash
cd "...TuRuta/Sin Framework"
python main.py
```

Para ejecutar el modelo con framework, corre en la terminal:
```bash
cd "...TuRuta/Con Framework"
python main.py
```

Al correrlo, cada versión te devolverá sus respectivas gráficas de correlación y del resultado final qu compara las predicciones con los valores reales de Ring. Para la versión sin framework, también se muestra una gráfica que muestra el descenso de MSE a lo de las épocas.

Igualmente, ambas devolverán en consola los valores finales de MSE y RMSE para el conjunto de entrenamiento y de pruebas, y mostrará una tabla para comparar los primeros 10 valores de Ring que predijo con los valores reales de Ring para ese abulón.

## Estado del reporte
Actualmente ya pude avanzarle más al reporte, pero vi qeu no abarco aún varias cosas que se solicitan, así que seguiré trabajando tanto en mi código como en el reporte para que cumplan con todos los puntos.
