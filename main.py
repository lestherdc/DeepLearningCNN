import yfinance as yf
import tensorflow as tf
import pandas as pd
from src.processor import DataProcessor
from src.brain_svj import SVJModel

#Configuracion inicial
SYMBOL = "PLTR"
tf.keras.config.enable_unsafe_deserialization() #Esto solo lo hare cuando yo mismo saque mis datos

#Funcion para niveles
def get_daily_levels(symbol):
    #Descargamos los ultimo 2 dias en velas diarias para ver el ayer
    daily = yf.download(symbol, period="3d", interval="1d", progress=False)

    #Limpieza multiIndex
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    #Eliminanos posibles filas vacias ( Fines de semanas y feriados donde el mercado no abre)
    daily = daily.dropna()

    if len(daily) < 2: return None

    #Ayer es la fila 0, Hoy es la fila 1
    levels = {
        "ayer_max": float(daily['High'].iloc[-2]),
        "ayer_min": float(daily['Low'].iloc[-2]),
        "hoy_max": float(daily['High'].iloc[-1]),
        "hoy_min": float(daily['Low'].iloc[-1]),
    }
    return levels

# 1. Cargar lo necesario
dl_model = tf.keras.models.load_model("models/cnn_lstm_v1.keras")
processor = DataProcessor(window_size=60)
processor.load_scaler("models/scaler_pltr.bin")

# 1.1 Obtener niveles historicos
niveles = get_daily_levels(SYMBOL)

# 2. Loop de mercado
raw_data = yf.download(SYMBOL, period="7d", interval="1m", progress=False)

#Limpieza de columnas
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

precio_actual = float(raw_data['Close'].iloc[-1])

# Predicción A: Deep Learning (Eventos)
X_live, _ = processor.create_dataset(raw_data, training=False)
dl_probs = dl_model.predict(X_live[-1:], verbose=0)[0]

# Predicción B: Matemática (Tendencia/Riesgo)
svj_results = SVJModel.calculate(raw_data)

#Reporte con niveles
print(f"\n" + "="*40)
print(f"ANÁLISIS EN VIVO PARA: {SYMBOL}")
print(f"Precio Actual: ${precio_actual:.2f}")
print("-" * 40)

# Alertas de niveles de ayer
if precio_actual >= niveles['ayer_max']:
    print(f"⚠️ ALERTA: Por encima del MÁXIMO DE AYER (${niveles['ayer_max']:.2f})")
elif precio_actual <= niveles['ayer_min']:
    print(f"⚠️ ALERTA: Por debajo del MÍNIMO DE AYER (${niveles['ayer_min']:.2f})")
else:
    distancia_max = niveles['ayer_max'] - precio_actual
    print(f"Info: El máximo de ayer está a ${distancia_max:.2f} de distancia.")

# Alertas de niveles de hoy
if precio_actual >= niveles['hoy_max'] * 0.999:
    print(f"📈 Testeando el MÁXIMO DE HOY (${niveles['hoy_max']:.2f})")

print("-" * 40)
print(f"DL -> Tocar Max: {dl_probs[0]*100:.1f}%")
print(f"SVJ -> P(Subida): {svj_results['p_up']:.1f}% | SCI: {svj_results['sci']:.1f}%")
print(f"DEBUG: {svj_results['debug']}")
print("="*40)