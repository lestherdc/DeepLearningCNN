import streamlit as st
import yfinance as yf
import tensorflow as tf
import pandas as pd
import os
from src.processor import DataProcessor
from src.brain_svj import SVJModel

# Configuración de la página
st.set_page_config(page_title="IA Stock Radar", layout="wide")

st.title("🎯 Radar de Visión Total con IA")
symbol = st.sidebar.selectbox("Selecciona Activo", ["TSLA", "PLTR", "AAPL", "AVGO"])

# --- Lógica de Carga (Simplificada para la web) ---
MODEL_PATH = f"models/{symbol}/model.keras"
SCALER_PATH = f"models/{symbol}/scaler.bin"

if os.path.exists(MODEL_PATH):
    # Cacheamos el modelo para que la web cargue rápido
    @st.cache_resource
    def load_ai_model(path):
        tf.keras.config.enable_unsafe_deserialization()
        return tf.keras.models.load_model(path)


    dl_model = load_ai_model(MODEL_PATH)
    processor = DataProcessor(window_size=60)
    processor.load_scaler(SCALER_PATH)

    # Descarga de datos
    raw_data = yf.download(symbol, period="20d", interval="5m", progress=False).dropna()
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    precio_actual = float(raw_data['Close'].iloc[-1])

    # Métricas principales en la parte superior
    col1, col2, col3 = st.columns(3)
    col1.metric("Precio Actual", f"${precio_actual:.2f}")

    # --- Cálculos de IA ---
    X_live, _ = processor.create_dataset(raw_data, training=False)
    dl_output = dl_model.predict(X_live[-1:], verbose=0)[0]
    p_up, p_down = dl_output[0] * 100, dl_output[1] * 100

    col2.metric("Sentimiento IA", "ALCISTA 🚀" if p_up > p_down else "BAJISTA 📉")
    col3.metric("Fuerza Up", f"{p_up:.1f}%")

    # --- Volatilidad y Objetivos ---
    recientes = raw_data['Close'].tail(20)
    volatilidad = recientes.pct_change().std() * precio_actual * 2

    # Mostrar en tablas bonitas
    st.subheader("Análisis de Objetivos (Probabilidad por Volatilidad)")
    # Aquí llamarías a tu función get_total_vision_levels...
    # (Suponiendo que ya la tienes importada)

    st.info(f"Volatilidad calculada (2 sigmas): ${volatilidad:.2f}")

else:
    st.error(f"No se encontró el modelo para {symbol}. Por favor, entrena el modelo primero.")