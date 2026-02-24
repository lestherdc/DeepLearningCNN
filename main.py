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
    #Descargue de 5 dias para evitar fines de semanas
    daily = yf.download(symbol, period="5d", interval="1d", progress=False)

    #Limpieza multiIndex
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    #Eliminanos posibles filas vacias ( Fines de semanas y feriados donde el mercado no abre)
    daily = daily.dropna()

    ayer = daily.iloc[-2]
    hoy = daily.iloc[-1]

    return {
        "ayer_max": float(ayer['High']),
        "ayer_min": float(ayer['Low']),
        "hoy_max": float(hoy['High']),
        "hoy_min": float(hoy['Low']),
    }

#Funcion para calcular RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 1. Cargar lo necesario
dl_model = tf.keras.models.load_model("models/cnn_lstm_5m_v1.keras") #Esto se cambio con nuevos modelos
processor = DataProcessor(window_size=60)
processor.load_scaler("models/scaler_pltr_5m.bin") #Esto se cambia con nuevos modelos

# 1.1 Obtener niveles historicos
niveles = get_daily_levels(SYMBOL)

# 2. Loop de mercado
raw_data = yf.download(SYMBOL, period="30d", interval="5m", progress=False)

#Limpieza de columnas
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

precio_actual = float(raw_data['Close'].iloc[-1])

# Predicción A: Deep Learning (Eventos)
X_live, _ = processor.create_dataset(raw_data, training=False)
dl_probs = dl_model.predict(X_live[-1:], verbose=0)[0]

prob_tocar_max = dl_probs[0]*100
prob_tocar_min = dl_probs[1]*100

# Predicción B: Matemática (Tendencia/Riesgo)
svj_results = SVJModel.calculate(raw_data)

#Calculos de RSI
raw_data['RSI'] = calculate_rsi(raw_data['Close'])
rsi_actual = float(raw_data['RSI'].iloc[-1])

#Reporte con niveles
print(f"\n" + "="*40)
print(f"ANÁLISIS EN VIVO PARA: {SYMBOL}")
print(f"Precio Actual: ${precio_actual:.2f}")
print("-" * 40)

# Alertas de niveles de hoy
if precio_actual >= niveles['hoy_max'] * 0.999:
    print(f"📈 Testeando el MÁXIMO DE HOY (${niveles['hoy_max']:.2f})")

#Bloque del RSI
print(f"RSI (Fuerza): {rsi_actual:.1f}")
if rsi_actual > 70:
    print("⚠️ SOBRECOMPRADO: El precio podría retroceder antes de seguir subiendo.")
elif rsi_actual < 30:
    print("📉 SOBREVENDIDO: Oportunidad de rebote al alza.")
else:
    print("✅ RSI Neutral: Hay espacio para moverse.")

print("-" * 40)
print(f"DL -> Prob. buscar Máximo de ayer: {prob_tocar_max:.1f}%")
print(f"DL -> Prob. buscar Mínimo de ayer: {prob_tocar_min:.1f}%")
if prob_tocar_max > prob_tocar_min:
    print(f"🎯 OBJETIVO: El modelo cree que irá por el MÁXIMO (${niveles['ayer_max']:.2f})")
else:
    print(f"🎯 OBJETIVO: El modelo cree que irá por el MÍNIMO (${niveles['ayer_min']:.2f})")
print(f"SVJ -> P(Subida): {svj_results['p_up']:.1f}% | SCI: {svj_results['sci']:.1f}%")
print(f"DEBUG: {svj_results['debug']}")
print("="*40)