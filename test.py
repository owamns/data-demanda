import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. Cargar los datos
data = pd.read_csv('data_clean.csv', parse_dates=['FECHA'], index_col='FECHA')

# 2. Generación de características
data['hora'] = data.index.hour
data['dia'] = data.index.day
data['semana'] = data.index.isocalendar().week
data['mes'] = data.index.month

# Desplazar los datos para obtener características de valores anteriores
for i in range(1, 4):  # Usar 3 valores anteriores
    data[f'prev_{i}'] = data['DEMANDA'].shift(i)

# Eliminar valores nulos generados por el desplazamiento
data.dropna(inplace=True)

# 3. División de datos
X = data.drop('DEMANDA', axis=1)
y = data['DEMANDA']

# Escalar los datos
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 4. Construcción del modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compilar el modelo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 5. Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=1, verbose=1)

# 6. Evaluación del modelo
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# Hacer predicciones
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Graficar las predicciones y los valores reales
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Valor Real')
plt.plot(y_pred, label='Predicción', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Tiempo')
plt.ylabel('Demanda')
plt.legend()
plt.show()
