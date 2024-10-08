import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Leer los datos desde el archivo CSV y asegurarse de que 'FECHA' se parsea como fechas
data = pd.read_csv('data.csv', parse_dates=['FECHA'])
data['FECHA'] = pd.to_datetime(data['FECHA'], dayfirst=True)
data.set_index('FECHA', inplace=True)

# Filtrar solo los datos correspondientes a los lunes
lunes_data = data[data.index.weekday == 0]

# Limpiar y preprocesar los datos
values = lunes_data['DEMANDA'].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values.reshape(-1, 1))

# Crear secuencias de datos para entrenamiento (p.ej., 30 días para predecir el siguiente día)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_values, SEQ_LENGTH)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definir el modelo de red neuronal (debe ser el mismo modelo que has entrenado)
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Cargar el modelo entrenado
model.load_weights('path_to_your_model_weights.h5')

# Crear secuencia para el 09/09/2024
last_sequence = scaled_values[-SEQ_LENGTH:].reshape((1, SEQ_LENGTH, 1))

# Hacer predicción
predicted_scaled = model.predict(last_sequence)
predicted = scaler.inverse_transform(predicted_scaled)

# Crear intervalos de 30 minutos para el 09/09/2024
intervalos = pd.date_range(start='2024-09-09 00:00', end='2024-09-09 23:30', freq='30T')

# Crear la gráfica
plt.figure(figsize=(12, 6))

# Graficar la demanda predicha
plt.plot(intervalos, predicted.flatten(), marker='o', linestyle='-', color='b', label='Predicción Demanda')

# Ajustes de la gráfica
plt.title('Predicción de Demanda para el 09/09/2024 en Intervalos de 30 Minutos')
plt.xlabel('Hora')
plt.ylabel('Demanda')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
