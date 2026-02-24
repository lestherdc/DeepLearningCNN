import yfinance as yf
import tensorflow as tf
from src.processor import DataProcessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

# 1. Configuración
SYMBOL = "PLTR"
WINDOW = 60
HORIZON = 30  # Ahora son 150 minutos a futuro (30 velas de 5m)

# 2. Descarga de datos masiva (velas de 5m)
print(f"Descargando datos de 5m para {SYMBOL}...")
# Nota: yfinance permite hasta 60 días de 5m, o más según la cuenta
data = yf.download(SYMBOL, period="60d", interval="5m")

# 3. Procesamiento
processor = DataProcessor(window_size=WINDOW, horizon=HORIZON)
X, y = processor.create_dataset(data, training=True)

# Guardamos el nuevo scaler específico para 5m
processor.save_scaler("models/scaler_pltr_5m.bin")
print("✅ Nuevo Scaler de 5m guardado.")

# 4. Arquitectura del Modelo (Optimizada para 5m)
model = Sequential([
    # Capa de Convolución para detectar patrones de velas
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW, X.shape[2])),
    MaxPooling1D(pool_size=2),

    # Capa LSTM para la memoria de la tendencia (5 horas de datos)
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),

    # Capa de salida (2 neuronas: Toca Max, Toca Min)
    Dense(32, activation='relu'),
    Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Entrenamiento
print("Entrenando cerebro de 5 minutos...")
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# 6. Guardar el nuevo modelo
model.save("models/cnn_lstm_5m_v1.keras")
print("✨ ¡Modelo de 5 minutos listo y guardado!")