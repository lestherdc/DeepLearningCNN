import yfinance as yf
import tensorflow as tf
import pandas as pd
from src.processor import DataProcessor
from src.brain_svj import SVJModel

# Configuración inicial
SYMBOL = "AAPL"
tf.keras.config.enable_unsafe_deserialization()

#Rutas automatizadas
MODEL_PATH = f"models/{SYMBOL}/model.keras"
SCALER_PATH = f"models/{SYMBOL}/scaler.bin"

# --- FUNCIONES DE NIVELES ---
def get_extended_levels(symbol):
    daily = yf.download(symbol, period="10d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)
    daily = daily.dropna()
    return {
        "ayer_min": float(daily['Low'].iloc[-2]),
        "antier_min": float(daily['Low'].iloc[-3])
    }


def get_historical_major_levels(symbol, precio_actual):
    hist = yf.download(symbol, period="2y", interval="1d", progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna()

    valles = hist['Low'][hist['Low'] == hist['Low'].rolling(window=5, center=True).min()].tolist()
    picos = hist['High'][hist['High'] == hist['High'].rolling(window=5, center=True).max()].tolist()

    # Soportes: de más cerca a más lejos (descendente)
    soportes = sorted(list(set([round(s, 2) for s in valles if s < precio_actual * 0.999])), reverse=True)
    # Resistencias: de más cerca a más lejos (ascendente)
    resistencias = sorted(list(set([round(r, 2) for r in picos if r > precio_actual * 1.001])))

    return {
        "soportes_h": soportes[:3],  # Los 3 niveles históricos más bajos
        "resistencia_h": resistencias[0] if resistencias else precio_actual * 1.05
    }


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


# 1. Cargar lo necesario
dl_model = tf.keras.models.load_model(MODEL_PATH)
processor = DataProcessor(window_size=60)
processor.load_scaler(SCALER_PATH)

# 2. Obtener Datos
raw_data = yf.download(SYMBOL, period="30d", interval="5m", progress=False)
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

precio_actual = float(raw_data['Close'].iloc[-1])
min_hoy_real = raw_data['Low'].tail(100).min()

# 2.1 Obtener niveles
niveles_c = get_extended_levels(SYMBOL)
niveles_h = get_historical_major_levels(SYMBOL, precio_actual)

# 3. Predicciones
X_live, _ = processor.create_dataset(raw_data, training=False)
dl_probs = dl_model.predict(X_live[-1:], verbose=0)[0]
p_subida, p_bajada = dl_probs[0] * 100, dl_probs[1] * 100
svj_results = SVJModel.calculate(raw_data)
raw_data['RSI'] = calculate_rsi(raw_data['Close'])
rsi_actual = float(raw_data['RSI'].iloc[-1])

# --- REPORTE RADAR 360° ---
print(f"\n" + "=" * 40)
print(f"RADAR DE OBJETIVOS: {SYMBOL} | Precio: ${precio_actual:.2f}")
print("-" * 40)

# Sección Superior
res_actual = niveles_h['resistencia_h']
print(f"🔼 TECHO (Resistencia): ${res_actual:.2f} | Prob: {p_subida:.1f}%")
print("-" * 20)

# Consolidar Soportes para el Radar
raw_levels = [
    {"nombre": "AYER", "precio": niveles_c['ayer_min']},
    {"nombre": "ANTIER", "precio": niveles_c['antier_min']}
]
for i, s_h in enumerate(niveles_h['soportes_h']):
    raw_levels.append({"nombre": f"HISTORICO {i + 1}", "precio": s_h})

# Ordenar por cercanía (descendente) y eliminar duplicados de precio
lista_soportes = sorted({v['precio']: v for v in raw_levels}.values(), key=lambda x: x['precio'], reverse=True)

objetivo_pendiente = None
precio_objetivo = None

for sop in lista_soportes:
    tocado = min_hoy_real <= sop['precio']
    status = "✅ TOCADO" if tocado else "⏳ PENDIENTE"

    # Probabilidad lógica: si está en tendencia bajista, los niveles pendientes tienen la prob. alta
    prob_nivel = p_bajada if not tocado else p_bajada * 0.8

    print(f"🔽 SOPORTE {sop['nombre']}: ${sop['precio']:.2f} | {status} | Prob: {prob_nivel:.1f}%")

    if not tocado and objetivo_pendiente is None:
        objetivo_pendiente = sop['nombre']
        precio_objetivo = sop['precio']

# 5. Conclusión Dinámica
print("-" * 40)
if p_bajada > p_subida:
    if precio_objetivo:
        print(f"🎯 PRÓXIMO OBJETIVO BAJISTA: ${precio_objetivo:.2f} ({objetivo_pendiente})")
    else:
        print(f"🎯 PRÓXIMO OBJETIVO BAJISTA: ${precio_actual * 0.98:.2f} (Soporte Dinámico)")
else:
    print(f"🎯 PRÓXIMO OBJETIVO ALCISTA: ${res_actual:.2f}")

print(f"RSI: {rsi_actual:.1f} | SCI: {svj_results['sci']:.1f}%")
print("=" * 40)