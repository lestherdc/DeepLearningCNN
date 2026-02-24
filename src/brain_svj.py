import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime, timezone


class SVJModel:
    @staticmethod
    def svj_log_likelihood(params, returns, dt):
        mu, kappa, theta, vol_vol, rho, lamb, mu_j, sig_j = params
        v = np.var(returns) + 1e-10
        log_lik = -np.sum(-0.5 * np.log(2 * np.pi * v) - 0.5 * (returns - mu * dt) ** 2 / v)
        return log_lik

    @staticmethod
    def calculate(df, beta_mercado=1.0):
        try:
            # 1. Limpieza de datos
            df_clean = df.copy()
            if isinstance(df_clean.columns, pd.MultiIndex):
                df_clean.columns = df_clean.columns.get_level_values(0)

            prices = pd.to_numeric(df_clean['Close'], errors='coerce').dropna().tail(500)

            if len(prices) < 10:
                return {"p_up": 50.0, "sci": 0.0, "status": "DATOS INSUFICIENTES", "debug": {"err": "Pocos datos"}}

            # 2. Preparación de retornos
            returns = np.log(prices / prices.shift(1)).dropna().values
            dt = 1 / 252

            # 3. Optimización
            initial_guess = [0.05, 2.0, 0.2, 0.3, -0.6, 0.1, 0.0, 0.2]
            bounds = [(-1, 1), (1e-2, 10), (1e-2, 1), (1e-2, 2), (-0.9, 0.9), (0, 5), (-1, 1), (1e-2, 1)]
            res = minimize(SVJModel.svj_log_likelihood, initial_guess, args=(returns, dt), method='L-BFGS-B',
                           bounds=bounds)

            # 4. Cálculo de tiempo
            last_candle_time = df.index[-1]
            now = datetime.now(timezone.utc) if last_candle_time.tzinfo else datetime.now()
            time_diff = abs((now.replace(tzinfo=None) - last_candle_time.replace(tzinfo=None)).total_seconds() / 60)

            # --- CALIBRACIÓN DE SCORES ---
            time_score = 30 * (1.0 if time_diff < 30 else max(0, 1 - (time_diff / 120)))
            error_score = max(0, 40 * (1 - (abs(res.fun) / 5000)))

            # MEJORA DE VOLUMEN: Usamos la penúltima vela (iloc[-2]) porque la última suele ser 0 en vivo
            avg_vol = df_clean['Volume'].tail(30).replace(0, np.nan).dropna().mean()

            # Intentamos tomar la vela cerrada (-2), si no hay suficientes datos, la actual (-1)
            vol_actual_val = df_clean['Volume'].iloc[-2] if len(df_clean) > 1 else df_clean['Volume'].iloc[-1]

            vol_ratio = min(1.0, vol_actual_val / (avg_vol * 0.4 + 1e-10))
            vol_score = 30 * vol_ratio

            sci = time_score + error_score + vol_score

            return {
                "p_up": (1 / (1 + np.exp(-res.x[0] * 15))) * 100,
                "sci": sci,
                "status": "CONFIABLE" if sci > 60 else "PRECAUCION",
                "debug": {
                    "time_gap": round(time_diff, 1),
                    "fit_err": round(float(res.fun), 1),
                    "vol": round(float(vol_score), 1)
                }
            }
        except Exception as e:
            return {"p_up": 50.0, "sci": 0.0, "status": "ERROR", "debug": {"err": str(e)[:20]}}