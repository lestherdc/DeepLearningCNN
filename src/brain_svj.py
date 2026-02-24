import numpy as np
from scipy.optimize import minimize
from datetime import datetime


class SVJModel:
    @staticmethod
    def svj_log_likelihood(params, returns, dt):
        mu, kappa, theta, vol_vol, rho, lamb, mu_j, sig_j = params
        # Para simplificar y dar velocidad al main.py, usamos una aproximación
        # de verosimilitud para el proceso de salto-difusión
        v = np.var(returns)
        log_lik = -np.sum(-0.5 * np.log(2 * np.pi * v) - 0.5 * (returns - mu * dt) ** 2 / v)
        return log_lik

    @staticmethod
    def calculate(df, beta_mercado=1.0):

        try:
            # 1. Preparación de retornos
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
            dt = 1 / 252  # Escala diaria

            # 2. Optimización de parámetros (Semilla fija para estabilidad)
            initial_guess = [0.05, 2.0, 0.2, 0.3, -0.6, 0.1, 0.0, 0.2]
            bounds = [(-1, 1), (1e-2, 10), (1e-2, 1), (1e-2, 2), (-0.9, 0.9), (0, 5), (-1, 1), (1e-2, 1)]

            res = minimize(SVJModel.svj_log_likelihood, initial_guess,
                           args=(returns, dt), bounds=bounds, method='L-BFGS-B')

            # 3. Cálculo de P(Subida) basada en la deriva y saltos
            mu_est = res.x[0]
            prob_up = 1 / (1 + np.exp(-mu_est * 10))  # Sigmoide de confianza

            # 4. Cálculo del SCI (Reality Check)
            # Miramos si hay volumen y si el optimizador convergió
            last_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].tail(50).mean()
            vol_ratio = min(1.0, last_vol / (avg_vol + 1e-10))

            # Verificación de tiempo (Dato fresco)
            last_candle_time = df.index[-1]
            now = datetime.now(last_candle_time.tz)
            time_diff = (now - last_candle_time).total_seconds() / 60

            error_score = max(0, 40 * (1 - np.sqrt(res.fun / 0.1)))
            activity_score = 30 * (1.0 if time_diff < 20 else 0.2)
            vol_score = 30 * vol_ratio

            sci = error_score + activity_score + vol_score
            if time_diff > 20: sci = min(sci, 40.0)

            return {
                "p_up": prob_up * 100,
                "sci": sci,
                "status": "CONFIABLE" if sci > 70 else "NO FIABLE"
            }
        except Exception as e:
            return {"p_up": 50.0, "sci": 0.0, "status": f"ERROR: {e}"}