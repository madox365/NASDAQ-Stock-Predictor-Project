import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

class StockDataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_full_data(self):
        """Carga stocks_full.csv"""
        path = f"{self.config['data']['processed_path']}/stocks_full.csv"
        df = pd.read_csv(path, parse_dates=['Date'])
        
        # Filtrar tickers si está especificado
        if self.config['data']['selected_tickers']:
            df = df[df['ticker'].isin(self.config['data']['selected_tickers'])]
        
        return df.sort_values(['ticker', 'Date'])
    
    def split_data(self, df):
        """Divide en train/val/test por fecha"""
        train_end = pd.to_datetime(self.config['data']['train_end'])
        val_end = pd.to_datetime(self.config['data']['val_end'])
        
        train = df[df['Date'] <= train_end]
        val = df[(df['Date'] > train_end) & (df['Date'] <= val_end)]
        test = df[df['Date'] > val_end]
        
        return train, val, test
    
    def save_splits(self, train, val, test):
        """Guarda los splits"""
        path = self.config['data']['processed_path']
        train.to_csv(f"{path}/train.csv", index=False)
        val.to_csv(f"{path}/val.csv", index=False)
        test.to_csv(f"{path}/test.csv", index=False)
        
        print(f"✓ Train: {len(train)} filas")
        print(f"✓ Val: {len(val)} filas")
        print(f"✓ Test: {len(test)} filas")