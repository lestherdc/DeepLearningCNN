import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib


class DataProcessor:
    def __init__(self, window_size=60, horizon=30):
        self.window_size = window_size
        self.horizon = horizon
        self.scaler = RobustScaler()

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)

    def prepare_features(self, df):
        # 1. Niveles del día anterior
        df = df.copy()
        # --- NUEVO: LIMPIEZA DE MULTIINDEX ---
        # Si las columnas son algo como ('High', 'PLTR'), nos quedamos solo con 'High'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Eliminamos cualquier espacio en blanco o nombres raros
        df.columns = [str(col).strip() for col in df.columns]
        # --------------------------------------

        df['Date'] = df.index.date

        # Cálculo de Volumen Direccional
        # Si el cierre es mayor que el anterior, el volumen es "positivo"
        df['vol_dir'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], -df['Volume'])

        # Media móvil del volumen direccional (acumulado de 10 velas)
        df['vol_force'] = df['vol_dir'].rolling(window=10).sum()

        # Ahora sí encontrará 'High' y 'Low' sin problema
        daily = df.groupby('Date').agg({'High': 'max', 'Low': 'min'}).shift(1)
        df = df.join(daily, on='Date', rsuffix='_prev')

        df['dist_max_prev'] = (df['High_prev'] - df['Close']) / df['Close']
        df['dist_min_prev'] = (df['Low_prev'] - df['Close']) / df['Close']

        # Volatilidad y Retornos
        df['returns'] = df['Close'].pct_change()
        df['vol_std'] = df['returns'].rolling(20).std()

        # RSI Simple para darle contexto de momentum a la CNN
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        return df.dropna()

    def create_dataset(self, df, training=True):
        df_feats = self.prepare_features(df)
        feature_cols = ['returns', 'vol_std', 'dist_max_prev', 'dist_min_prev', 'rsi', 'Volume']

        # Escalado de datos
        if training:
            scaled_data = self.scaler.fit_transform(df_feats[feature_cols])
        else:
            scaled_data = self.scaler.transform(df_feats[feature_cols])

        X, y = [], []
        for i in range(self.window_size, len(df_feats) - self.horizon):
            X.append(scaled_data[i - self.window_size:i])

            # Etiquetado: ¿Tocará el nivel en el horizonte futuro?
            future_high = df_feats['High'].iloc[i: i + self.horizon].max()
            future_low = df_feats['Low'].iloc[i: i + self.horizon].min()
            max_prev = df_feats['High_prev'].iloc[i]
            min_prev = df_feats['Low_prev'].iloc[i]

            toca_max = 1 if future_high >= max_prev else 0
            toca_min = 1 if future_low <= min_prev else 0
            y.append([toca_max, toca_min])

        return np.array(X), np.array(y)