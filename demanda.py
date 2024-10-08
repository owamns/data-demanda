import pandas as pd
import matplotlib.pyplot as plt

# Leer los datos desde el archivo CSV y asegurarse de que 'FECHA' se parsea como fechas
data = pd.read_csv('data.csv', parse_dates=['FECHA'])

# Convertir la columna FECHA a un índice de tipo datetime con dayfirst=True
data['FECHA'] = pd.to_datetime(data['FECHA'], dayfirst=True)
data.set_index('FECHA', inplace=True)

# Agrupar los datos por semana
semanas = data['DEMANDA'].resample('W')

# Crear la figura para la gráfica
plt.figure(figsize=(10, 6))

# Iterar sobre cada semana y graficar en la misma figura
for semana, datos in semanas:
    # Obtener los datos de demanda para la semana
    demanda_semanal = datos.reset_index()
    
    # Crear un eje X basado en los intervalos de 30 minutos
    eje_x = range(len(demanda_semanal))
    
    # Graficar la demanda de la semana actual con su propio label
    plt.plot(eje_x, demanda_semanal['DEMANDA'], marker='o', linestyle='-', label=f'Semana que termina en {semana.date()}')

# Ajustes de la gráfica
plt.title('Demanda Semanal Superpuesta por Intervalos de 30 Minutos')
plt.xlabel('Intervalo de 30 Minutos')
plt.ylabel('Demanda')
plt.xticks(rotation=45)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Semanas')
plt.grid(True)
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------


# Leer los datos desde el archivo CSV y asegurarse de que 'FECHA' se parsea como fechas
data = pd.read_csv('data.csv', parse_dates=['FECHA'])

# Convertir la columna FECHA a un índice de tipo datetime con dayfirst=True
data['FECHA'] = pd.to_datetime(data['FECHA'], dayfirst=True)
data.set_index('FECHA', inplace=True)

# Filtrar solo los datos correspondientes a los lunes
lunes_data = data[data.index.weekday == 0]

# Agrupar los datos de los lunes por semana
semanas_lunes = lunes_data['DEMANDA'].resample('W')

# Crear la figura para la gráfica
plt.figure(figsize=(10, 6))

# Iterar sobre cada lunes por semana y graficar en la misma figura
for semana, datos in semanas_lunes:
    # Obtener los datos de demanda para cada lunes de la semana
    demanda_lunes = datos.reset_index()
    
    # Crear un eje X basado en los intervalos de 30 minutos del lunes
    eje_x = range(len(demanda_lunes))
    
    # Graficar la demanda del lunes actual con su propio label
    plt.plot(eje_x, demanda_lunes['DEMANDA'], marker='o', linestyle='-')

# Ajustes de la gráfica
plt.title('Demanda de los Lunes Superpuesta por Intervalos de 30 Minutos')
plt.xlabel('Intervalo de 30 Minutos (Lunes)')
plt.ylabel('Demanda')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------

# Leer los datos desde el archivo CSV y asegurarse de que 'FECHA' se parsea como fechas
data = pd.read_csv('data.csv', parse_dates=['FECHA'])

# Convertir la columna FECHA a un índice de tipo datetime con dayfirst=True
data['FECHA'] = pd.to_datetime(data['FECHA'], dayfirst=True)
data.set_index('FECHA', inplace=True)

# Filtrar solo los datos correspondientes a los lunes
lunes_data = data[data.index.weekday == 0]

# Agrupar los datos de los lunes por semana y calcular la media de cada semana
semanas_lunes = lunes_data['DEMANDA'].resample('W').mean()

# Calcular percentiles (p.ej., el percentil 10)
percentil_10 = semanas_lunes.quantile(0.30)

# Calcular el rango intercuartílico (IQR)
Q1 = semanas_lunes.quantile(0.20)
Q3 = semanas_lunes.quantile(0.80)
IQR = Q3 - Q1

# Determinar el umbral basado en el percentil 10 y el IQR
umbral_percentil = percentil_10
umbral_iqr = Q1 - 1.5 * IQR

# Usar el umbral más conservador (el más bajo)
umbral = min(umbral_percentil, umbral_iqr)

# Filtrar las semanas que están por encima del umbral
semanas_filtradas = semanas_lunes[semanas_lunes >= umbral]

# Filtrar los datos originales basados en las semanas filtradas
lunes_filtrados = lunes_data[lunes_data.index.to_period('W').astype(str).isin(semanas_filtradas.index.to_period('W').astype(str))]

lunes_filtrados.to_csv('data_clean.csv')

# Agrupar los datos filtrados por semana
semanas_lunes_filtradas = lunes_filtrados['DEMANDA'].resample('W')

# Crear la figura para la gráfica
plt.figure(figsize=(10, 6))

# Iterar sobre cada semana filtrada y graficar en la misma figura
for semana, datos in semanas_lunes_filtradas:
    # Obtener los datos de demanda para cada lunes filtrado
    demanda_lunes = datos.reset_index()
    
    # Crear un eje X basado en los intervalos de 30 minutos del lunes
    eje_x = range(len(demanda_lunes))
    
    # Graficar la demanda del lunes actual con su propio label
    plt.plot(eje_x, demanda_lunes['DEMANDA'], marker='o', linestyle='-', label=f'Lunes de la semana que termina en {semana.date()}')

# Ajustes de la gráfica
plt.title('Demanda de los Lunes Filtrada y Superpuesta por Intervalos de 30 Minutos')
plt.xlabel('Intervalo de 30 Minutos (Lunes)')
plt.ylabel('Demanda')
plt.xticks(rotation=45)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Semanas')
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar umbrales para referencia
print(f'Umbral basado en percentil 10: {umbral_percentil:.2f}')
print(f'Umbral basado en IQR: {umbral_iqr:.2f}')
print(f'Umbral final utilizado: {umbral:.2f}')


#----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv', parse_dates=['FECHA'])
data['FECHA'] = pd.to_datetime(data['FECHA'], dayfirst=True)
data.set_index('FECHA', inplace=True)
lunes_data = data[data.index.weekday == 0]
values = lunes_data['DEMANDA'].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values.reshape(-1, 1))

# Crear secuencias de datos para entrenamiento
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

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definir el modelo con Dropout y Early Stopping
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Graficar las predicciones y los valores reales
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Valor Real')
plt.plot(y_pred_rescaled, label='Predicción', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Tiempo')
plt.ylabel('Demanda')
plt.legend()
plt.show()(X_train, y_train, epochs=50, batch_size=1, verbose=1, validation_split=0.1, callbacks=[early_stopping])

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Graficar las predicciones y los valores reales
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Valor Real')
plt.plot(y_pred_rescaled, label='Predicción', linestyle='--')
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Tiempo')
plt.ylabel('Demanda')
plt.legend()
plt.show()
