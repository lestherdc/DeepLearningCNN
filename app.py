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

# ... (código anterior)

# Descarga de datos
raw_data = yf.download(symbol, period="20d", interval="5m", progress=False)

# 1. Validar si el DataFrame está vacío
if raw_data.empty:
    st.error(f"No se pudieron obtener datos para {symbol}. Verifica el símbolo o intenta más tarde.")
else:
    # 2. Limpiar MultiIndex si es necesario
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # 3. Eliminar valores nulos
    raw_data = raw_data.dropna()

    # 4. Verificar de nuevo tras la limpieza
    if len(raw_data) > 0:
        precio_actual = float(raw_data['Close'].iloc[-1])

        # --- Continuar con el resto de la lógica ---
        st.metric("Precio Actual", f"${precio_actual:.2f}")
    else:
        st.warning("El mercado podría estar cerrado o no hay suficientes velas de 5m.")


# --- FUNCIÓN DE NIVELES (Asegúrate de que esté definida o importada) ---
def get_total_vision_levels(symbol, precio_actual, daily):
    # --- 1. SIEMPRE INICIALIZAR PRIMERO ---
    soportes = []
    resistencias = []
    etiquetas = {}

    # Validar que daily tenga datos para evitar que los bucles se salten
    if daily.empty:
        return soportes, resistencias, etiquetas

    # 2. Tu lógica de búsqueda (Asegúrate de que 'max_hoy' y 'min_hoy' estén definidos)
    # Ejemplo de estructura de búsqueda:
    for i in range(1, len(daily)):
        idx = -(i + 1)
        # ... (tu lógica de detección de niveles vírgenes) ...
        # d_high = round(float(daily['High'].iloc[idx]), 2)
        # if d_high > precio_actual:
        #     resistencias.append(d_high)
        #     etiquetas[d_high] = "Máximo histórico"

    # --- 3. RETORNO SEGURO ---
    # Si no encontró nada, devolverá listas vacías [], lo cual evita el NameError
    resis_finales = sorted(list(set(resistencias)))[:3]
    sops_finales = sorted(list(set(soportes)), reverse=True)[:3]

    return sops_finales, resis_finales, etiquetas


# --- DENTRO DE LA LÓGICA DE STREAMLIT (después de validar raw_data) ---

# 1. Obtener Niveles Históricos
# Descargamos daily por separado para la visión de 6 meses
daily_data = yf.download(symbol, period="180d", interval="1d", progress=False)
if isinstance(daily_data.columns, pd.MultiIndex):
    daily_data.columns = daily_data.columns.get_level_values(0)

soportes, resistencias, etiquetas = get_total_vision_levels(symbol, precio_actual, daily_data)

# 2. Cálculo de Volatilidad (2 sigmas)
recientes = raw_data['Close'].tail(20)
volatilidad_2s = recientes.pct_change().std() * precio_actual * 2

# 3. Construcción de Tablas de Objetivos
st.subheader("🚀 Radar de Objetivos Históricos")

# --- TABLA DE RESISTENCIAS ---
st.markdown("### 🔼 Resistencias Pendientes (Techos)")
if resistencias:
    data_res = []
    for r in resistencias:
        dist_pts = abs(r - precio_actual)
        dist_pct = (dist_pts / precio_actual) * 100
        factor_v = dist_pts / max(volatilidad_2s, 0.01)
        # p_up viene de tu modelo dl_model.predict
        prob = max(p_up / (1 + factor_v), 1.0)

        data_res.append({
            "Nivel ($)": f"{r:.2f}",
            "Probabilidad": f"{prob:.1f}%",
            "Distancia": f"+{dist_pct:.2f}%",
            "Origen": etiquetas[r]
        })
    st.table(pd.DataFrame(data_res))
else:
    st.success("Subida Libre: No hay techos históricos cercanos.")

# --- TABLA DE SOPORTES ---
st.markdown("### 🔽 Soportes Pendientes (Suelos)")
if soportes:
    data_sop = []
    for s in soportes:
        dist_pts = abs(precio_actual - s)
        dist_pct = (dist_pts / precio_actual) * 100
        factor_v = dist_pts / max(volatilidad_2s, 0.01)
        # p_down viene de tu modelo dl_model.predict
        prob = max(p_down / (1 + factor_v), 1.0)

        data_sop.append({
            "Nivel ($)": f"{s:.2f}",
            "Probabilidad": f"{prob:.1f}%",
            "Distancia": f"-{dist_pct:.2f}%",
            "Origen": etiquetas[s]
        })
    st.table(pd.DataFrame(data_sop))
else:
    st.warning("Caída Libre: No hay suelos detectados en 6 meses.")