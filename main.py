import yfinance as yf
import tensorflow as tf
import pandas as pd
from src.processor import DataProcessor
from src.brain_svj import SVJModel

#Configuracion inicial
SYMBOL = "PLTR"
tf.keras.config.enable_unsafe_deserialization() #Esto solo lo hare cuando yo mismo saque mis datos

#Funcion para niveles
def get_extended_levels(symbol):
    #Descargue de 5 dias para evitar fines de semanas
    daily = yf.download(symbol, period="10d", interval="1d", progress=False)

    #Limpieza multiIndex
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    #Eliminanos posibles filas vacias ( Fines de semanas y feriados donde el mercado no abre)
    daily = daily.dropna()

    #Creacion de mapa de niveles
    levels = {
        "ayer_max": float(daily['High'].iloc[-2]),
        "ayer_min": float(daily['Low'].iloc[-2]),
        "antier_max": float(daily['High'].iloc[-3]),
        "antier_min": float(daily['Low'].iloc[-3]),
        "hoy_max": float(daily['High'].iloc[-1]),
        "hoy_min": float(daily['Low'].iloc[-1])
    }

    return levels


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
niveles = get_extended_levels(SYMBOL)

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


# Obtenemos el máximo y mínimo alcanzado HOY (desde la apertura)
max_hoy_real = raw_data['High'].tail(100).max() # Ajustado a la sesión actual
min_hoy_real = raw_data['Low'].tail(100).min()

ya_toco_min_ayer = min_hoy_real <= niveles['ayer_min']
ya_toco_max_ayer = max_hoy_real >= niveles['ayer_max']

# --- REPORTE LIMPIO  ---
print(f"\n" + "="*40)
print(f"ANÁLISIS EN VIVO PARA: {SYMBOL}")
print(f"Precio Actual: ${precio_actual:.2f}")
print("-" * 40)

# 1. Definición dinámica de objetivos
if precio_actual <= niveles['ayer_min']:
    # Estamos por debajo o en el mínimo de ayer -> Miramos hacia abajo (Antier) o rebote
    objetivo_baja = niveles['antier_min']
    objetivo_alta = niveles['ayer_min'] # El soporte roto ahora es resistencia
    status_contexto = "📉 Mínimo de ayer perforado."
else:
    # Estamos dentro del rango -> Los objetivos son los de ayer
    objetivo_baja = niveles['ayer_min']
    objetivo_alta = niveles['ayer_max']
    status_contexto = "⚖️ Cotizando en rango de ayer."

print(f"Estatus: {status_contexto}")

# 2. Lógica de Predicción del DL (Sin ruido)
print(f"Confianza: Máx {prob_tocar_max:.1f}% | Mín {prob_tocar_min:.1f}%")

# Elegimos qué imprimir basándonos en la probabilidad predominante y el precio actual
if prob_tocar_max > prob_tocar_min:
    direccion = "ALCISTA" if objetivo_alta > precio_actual else "RECUPERACIÓN"
    print(f"🎯 OBJETIVO {direccion}: ${objetivo_alta:.2f}")
else:
    # Si ya tocamos el mínimo de ayer, el próximo mínimo es el de antier
    target_final = objetivo_baja if ya_toco_min_ayer else niveles['ayer_min']
    direccion = "BAJISTA" if target_final < precio_actual else "LATERAL"
    print(f"🎯 OBJETIVO {direccion}: ${target_final:.2f}")

# 3. Bloque de indicadores
print("-" * 40)
print(f"RSI: {rsi_actual:.1f} | SCI: {svj_results['sci']:.1f}%")

if rsi_actual > 70: print("⚠️ SOBRECOMPRADO")
elif rsi_actual < 30: print("📉 SOBREVENDIDO")

print(f"SVJ P(Subida): {svj_results['p_up']:.1f}%")
print(f"DEBUG: {svj_results['debug']}")
print("="*40)