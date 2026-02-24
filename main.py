# main.py simplificado
import yfinance as yf
import tensorflow as tf
from src.processor import DataProcessor
from src.brain_dl import create_model # O load_model directamente
from src.brain_svj import SVJModel

tf.keras.config.enable_unsafe_deserialization() #Esto solo lo hare cuando yo mismo saque mis datos


# 1. Cargar lo necesario
dl_model = tf.keras.models.load_model("models/cnn_lstm_v1.keras")
processor = DataProcessor(window_size=60)
processor.load_scaler("models/scaler_pltr.bin")

# 2. Loop de mercado
raw_data = yf.download("PLTR", period="5d", interval="1m")

# Predicción A: Deep Learning (Eventos)
X_live, _ = processor.create_dataset(raw_data, training=False)
dl_probs = dl_model.predict(X_live[-1:], verbose=0)[0]

# Predicción B: Matemática (Tendencia/Riesgo)
svj_results = SVJModel.calculate(raw_data)

print(f"DL -> Tocar Max: {dl_probs[0]*100:.1f}%")
print(f"SVJ -> P(Subida): {svj_results['p_up']:.1f}% | SCI: {svj_results['sci']:.1f}%")