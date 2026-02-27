import yfinance as yf
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from src.processor import DataProcessor
from src.brain_svj import SVJModel

# --- CONFIGURACIÓN ---
SYMBOL = "TSLA"
tf.keras.config.enable_unsafe_deserialization()
MODEL_PATH = f"models/{SYMBOL}/model.keras"
SCALER_PATH = f"models/{SYMBOL}/scaler.bin"


def get_total_vision_levels(symbol, precio_actual):
    """Obtiene objetivos reales filtrando por testeo histórico (Visión 6 meses)"""
    # 1. Datos diarios de largo plazo para niveles históricos
    # Aumentamos a 180 días para tener "visión" real de soportes viejos
    daily = yf.download(symbol, period="180d", interval="1d", progress=False).dropna()
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = daily.columns.get_level_values(0)

    # Datos de hoy para saber el rango ya recorrido
    intraday = yf.download(symbol, period="1d", interval="5m", progress=False).dropna()
    if not intraday.empty:
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        max_hoy = intraday['High'].max()
        min_hoy = intraday['Low'].min()
    else:
        max_hoy = precio_actual
        min_hoy = precio_actual

    resis_finales = []
    sops_finales = []
    etiquetas = {}

    # 3. Escaneo de niveles (Ayer hacia atrás 180 días)
    for i in range(1, len(daily)):
        idx = -(i + 1)
        if abs(idx) >= len(daily): break

        fecha = daily.index[idx]
        d_high = round(float(daily['High'].iloc[idx]), 2)
        d_low = round(float(daily['Low'].iloc[idx]), 2)

        # --- RESISTENCIAS ---
        if d_high > precio_actual and d_high > max_hoy:
            # Solo la agregamos si no ha sido testeada por precios posteriores
            # (Simplificado: verificamos si algún cierre diario posterior la superó)
            posteriores = daily.iloc[idx + 1:]
            if posteriores['High'].max() < d_high:
                if d_high not in etiquetas:
                    resis_finales.append(d_high)
                    etiquetas[d_high] = f"Máximo del {fecha.strftime('%d/%m/%y')}"

        # --- SOPORTES ---
        if d_low < precio_actual and d_low < min_hoy:
            # Solo si ningún mínimo posterior lo rompió
            posteriores = daily.iloc[idx + 1:]
            if posteriores['Low'].min() > d_low:
                if d_low not in etiquetas:
                    sops_finales.append(d_low)
                    etiquetas[d_low] = f"Mínimo del {fecha.strftime('%d/%m/%y')}"

    # Ordenar por proximidad al precio actual
    resis = sorted(resis_finales)[:3]  # Los 3 techos más cercanos
    sops = sorted(sops_finales, reverse=True)[:3]  # Los 3 suelos más cercanos

    return sops, resis, etiquetas


# --- EJECUCIÓN Y CÁLCULOS ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Modelo no encontrado.")
    exit()

dl_model = tf.keras.models.load_model(MODEL_PATH)
processor = DataProcessor(window_size=60)
processor.load_scaler(SCALER_PATH)

raw_data = yf.download(SYMBOL, period="20d", interval="5m", progress=False).dropna()
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

precio_actual = float(raw_data['Close'].iloc[-1])
soportes, resistencias, etiquetas = get_total_vision_levels(SYMBOL, precio_actual)

# IA y Métricas
X_live, _ = processor.create_dataset(raw_data, training=False)
dl_output = dl_model.predict(X_live[-1:], verbose=0)[0]
p_subida, p_bajada = dl_output[0] * 100, dl_output[1] * 100
svj_results = SVJModel.calculate(raw_data)

# --- REPORTE CON VISIÓN AMPLIADA ---
print(f"\n" + "█" * 75)
print(f"🎯 RADAR DE VISIÓN TOTAL: {SYMBOL} | PRECIO: ${precio_actual:.2f}")
print("█" * 75)

print("🔼 RESISTENCIAS PENDIENTES (Techos históricos vírgenes):")
if not resistencias:
    print("   🚀 SUBIDA LIBRE: No hay máximos previos en los últimos 6 meses.")
else:
    for i, r in enumerate(resistencias):
        dist = ((r - precio_actual) / precio_actual) * 100
        # Probabilidad: Fuerza IA ajustada por la distancia (penaliza 5% por cada 1% de distancia)
        prob_obj = p_subida * (1 - (dist / 15))
        print(f"   OBJ {i + 1}: ${r:7.2f} | Prob: {max(prob_obj, 1.0):5.1f}% | (+{dist:5.2f}%) | {etiquetas[r]}")

print("-" * 60)

print("🔽 SOPORTES PENDIENTES (Suelos históricos vírgenes):")
if not soportes:
    print("   ⚠️ CAÍDA LIBRE: No hay soportes detectados en el historial reciente.")
else:
    for i, s in enumerate(soportes):
        dist = ((precio_actual - s) / precio_actual) * 100
        prob_obj = p_bajada * (1 - (dist / 15))
        print(f"   OBJ {i + 1}: ${s:7.2f} | Prob: {max(prob_obj, 1.0):5.1f}% | (-{dist:5.2f}%) | {etiquetas[s]}")

print("-" * 75)
print(f"SENTIMIENTO IA: {'ALCISTA 🚀' if p_subida > p_bajada else 'BAJISTA 📉'} | RSI: {svj_results.get('rsi', 0):.1f}")
print(f"SCI: {svj_results['sci']:.1f}% | SVJ P(Up): {svj_results['p_up']:.1f}%")
print("█" * 75)