import yfinance as yf
from src.processor import DataProcessor
from src.brain_dl import create_model

# 1. Cargar datos (Ejemplo con PLTR)
print("Bajando datos históricos...")
raw_data = yf.download("PLTR", period="2y", interval="1h") # Recomendado 1h o 15m

# 2. Procesar
processor = DataProcessor(window_size=60, horizon=20)
X, y = processor.create_dataset(raw_data)

# 2.1 Guardado de "procesador"
processor.save_scaler("models/scaler_pltr.bin")
print("Escalador guardado.")

# 3. Dividir y Entrenar
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = create_model(input_shape=(X.shape[1], X.shape[2]))
print("Entrenando modelo...")

model.fit(X_train, y_train,
          epochs=30,
          batch_size=32,
          validation_data=(X_test, y_test),
          verbose=1)

# 4. Guardar
model.save("models/cnn_lstm_v1.keras")
print("Modelo guardado en models/cnn_lstm_v1.keras")