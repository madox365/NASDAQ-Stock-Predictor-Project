import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from tqdm import tqdm

class FeatureEngineering:
    def __init__(self, config):
        self.config = config
        
    def add_technical_indicators(self, df):
        """Agrega indicadores técnicos por ticker - VERSIÓN OPTIMIZADA"""
        print(f"\nCalculando indicadores técnicos...")
        print(f"  Tickers: {df['ticker'].nunique()}")
        print(f"  Filas totales: {len(df):,}")
        
        # Ordenar una sola vez
        df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        # Usar groupby en lugar de bucle (mucho más eficiente)
        grouped = df.groupby('ticker', group_keys=False)
        
        # Calcular indicadores por grupo
        print("  Calculando SMA...")
        if 'SMA_20' in self.config['features']['technical_indicators']:
            df['SMA_20'] = grouped['Close'].transform(
                lambda x: SMAIndicator(x, window=20).sma_indicator()
            )
        
        if 'SMA_50' in self.config['features']['technical_indicators']:
            df['SMA_50'] = grouped['Close'].transform(
                lambda x: SMAIndicator(x, window=50).sma_indicator()
            )
        
        print("  Calculando EMA...")
        if 'EMA_12' in self.config['features']['technical_indicators']:
            df['EMA_12'] = grouped['Close'].transform(
                lambda x: EMAIndicator(x, window=12).ema_indicator()
            )
        
        print("  Calculando RSI...")
        if 'RSI_14' in self.config['features']['technical_indicators']:
            df['RSI_14'] = grouped['Close'].transform(
                lambda x: RSIIndicator(x, window=14).rsi()
            )
        
        print("  Calculando MACD...")
        if 'MACD' in self.config['features']['technical_indicators']:
            df['MACD'] = grouped['Close'].transform(
                lambda x: MACD(x).macd()
            )
            df['MACD_signal'] = grouped['Close'].transform(
                lambda x: MACD(x).macd_signal()
            )
        
        print("  Calculando Bollinger Bands...")
        if 'Bollinger_Bands' in self.config['features']['technical_indicators']:
            df['BB_high'] = grouped['Close'].transform(
                lambda x: BollingerBands(x).bollinger_hband()
            )
            df['BB_low'] = grouped['Close'].transform(
                lambda x: BollingerBands(x).bollinger_lband()
            )
        
        # Eliminar NaN generados por indicadores
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        print(f"\n  ✓ Indicadores calculados")
        print(f"    Filas finales: {len(df):,} (eliminadas {dropped_rows:,} con NaN)")
        
        return df
    
    def create_sequences(self, df, feature_cols, target_col='Close'):
        """Crea secuencias para modelos temporales - VERSIÓN ULTRA OPTIMIZADA"""
        lookback = self.config['features']['lookback_window']
        horizon = self.config['features']['prediction_horizon']
        
        print(f"\nCreando secuencias:")
        print(f"  Lookback: {lookback} días")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Horizon: {horizon} día(s)")
        
        # PRE-AGRUPAR UNA SOLA VEZ (clave para velocidad)
        print("  Pre-agrupando datos...")
        df_sorted = df.sort_values(['ticker', 'Date']).reset_index(drop=True)
        grouped = df_sorted.groupby('ticker', sort=False)
        
        # Pre-calcular tamaño total eficientemente
        ticker_lengths = grouped.size()
        valid_tickers = ticker_lengths[ticker_lengths > lookback + horizon]
        total_sequences = (valid_tickers - lookback - horizon + 1).sum()
        
        print(f"  Total secuencias estimadas: {total_sequences:,}")
        print(f"  Tickers válidos: {len(valid_tickers)}/{len(ticker_lengths)}")
        
        # Pre-asignar arrays (clave para velocidad y RAM)
        X = np.zeros((total_sequences, lookback, len(feature_cols)), dtype=np.float32)
        y = np.zeros(total_sequences, dtype=np.float32)
        tickers_list = []
        dates_list = []
        
        # Procesar por ticker con barra de progreso
        idx = 0
        print("  Generando secuencias...")
        for ticker, group in tqdm(grouped, desc="  Procesando tickers", total=len(grouped)):
            if len(group) <= lookback + horizon:
                continue
            
            # El grupo ya está ordenado por Date
            values = group[feature_cols].values.astype(np.float32)
            targets = group[target_col].values.astype(np.float32)
            dates = group['Date'].values
            
            n_sequences = len(values) - lookback - horizon + 1
            
            # Bucle interno optimizado
            for i in range(n_sequences):
                X[idx] = values[i:i+lookback]
                y[idx] = targets[i+lookback+horizon-1]
                tickers_list.append(ticker)
                dates_list.append(dates[i+lookback+horizon-1])
                idx += 1
        
        # Recortar al tamaño exacto
        X = X[:idx]
        y = y[:idx]
        
        print(f"\n  ✓ Secuencias creadas: {len(X):,}")
        print(f"    Shape: {X.shape}")
        print(f"    RAM: {X.nbytes / 1e9:.2f} GB")
        print(f"    Tiempo aproximado: 1-3 minutos")
        
        return X, y, tickers_list, dates_list