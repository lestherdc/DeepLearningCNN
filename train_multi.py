import os
import yfinance as yf
import pandas as pd
from src.processor import DataProcessor
import tensorflow as tf

# ---- COnfiguracion -----
ACCIONES = ["PLTR", "TSLA", "AAPL", "AVGO"]
DIAS_HISTORIA = "60d"
INTERVALO = "5m"
WINDOW_SIZE = 60


def train_all_models():
    for symbol in ACCIONES:
        print(f"\n" + "=" * 50)
        print(f"🚀 INICIANDO ENTRENAMIENTO PARA: {symbol}")
        print("=" * 50)

        # 1. Crear carpeta para el modelo si no existe
        model_dir = f"models/{symbol}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 2. Descargar Datos
        print(f"📥 Descargando datos de {symbol}...")
        df = yf.download(symbol, period=DIAS_HISTORIA, interval=INTERVALO, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        # 3. Procesamiento y Scaler Individual
        processor = DataProcessor(window_size=WINDOW_SIZE)
        X, y = processor.create_dataset(df, training=True)

        # Guardar el scaler específico de esta acción
        scaler_path = f"{model_dir}/scaler.bin"
        processor.save_scaler(scaler_path)
        print(f"✅ Scaler guardado en: {scaler_path}")

        # 4. Definir Arquitectura del Modelo (CNN + LSTM)
        # Usamos la misma lógica que ya te funcionó para PLTR
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(WINDOW_SIZE, X.shape[2])),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation='sigmoid')  # [Prob_Max, Prob_Min]
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 5. Entrenamiento
        print(f"🧠 Entrenando cerebro para {symbol}...")
        model.fit(X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

        # 6. Guardar Modelo
        model_path = f"{model_dir}/model.keras"
        model.save(model_path)
        print(f"💾 Modelo {symbol} guardado exitosamente!")


if __name__ == "__main__":
    train_all_models()