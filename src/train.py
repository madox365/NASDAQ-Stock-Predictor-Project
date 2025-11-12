import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from datetime import datetime

# Importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, set_seed, create_directories
from src.data_loader import StockDataLoader
from src.features import FeatureEngineering
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.tcn import TCNModel
from src.models.tft import TFTModel

class TrainingPipeline:
    def __init__(self, config_path='config.yaml'):
        """Inicializa el pipeline de entrenamiento"""
        self.config = load_config(config_path)
        set_seed(self.config['training']['random_seed'])
        create_directories(self.config)
        
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_and_prepare_data(self):
        """Carga y prepara los datos"""
        print("\n" + "="*70)
        print("PASO 1: CARGANDO Y PREPARANDO DATOS")
        print("="*70)
        
        # Cargar datos
        loader = StockDataLoader(self.config)
        
        # Opción 1: Usar splits existentes
        processed_path = self.config['data']['processed_path']
        
        if os.path.exists(f"{processed_path}/train.csv"):
            print("✓ Cargando splits existentes...")
            train_df = pd.read_csv(f"{processed_path}/train.csv", parse_dates=['Date'])
            val_df = pd.read_csv(f"{processed_path}/val.csv", parse_dates=['Date'])
            test_df = pd.read_csv(f"{processed_path}/test.csv", parse_dates=['Date'])
        else:
            print("✓ Creando nuevos splits...")
            full_df = loader.load_full_data()
            train_df, val_df, test_df = loader.split_data(full_df)
            loader.save_splits(train_df, val_df, test_df)
        
        print(f"\nTrain: {len(train_df):,} filas, {train_df['ticker'].nunique()} tickers")
        print(f"Val:   {len(val_df):,} filas, {val_df['ticker'].nunique()} tickers")
        print(f"Test:  {len(test_df):,} filas, {test_df['ticker'].nunique()} tickers")
        
        return train_df, val_df, test_df
    
    def engineer_features(self, train_df, val_df, test_df):
        """Agrega indicadores técnicos"""
        print("\n" + "="*70)
        print("PASO 2: INGENIERÍA DE CARACTERÍSTICAS")
        print("="*70)
        
        feature_eng = FeatureEngineering(self.config)
        
        print("✓ Calculando indicadores técnicos (puede tardar)...")
        train_df = feature_eng.add_technical_indicators(train_df)
        val_df = feature_eng.add_technical_indicators(val_df)
        test_df = feature_eng.add_technical_indicators(test_df)
        
        print(f"Train después de features: {len(train_df):,} filas")
        print(f"Val después de features:   {len(val_df):,} filas")
        print(f"Test después de features:  {len(test_df):,} filas")
        
        return train_df, val_df, test_df
    
    def create_sequences(self, train_df, val_df, test_df):
        """Crea secuencias temporales"""
        print("\n" + "="*70)
        print("PASO 3: CREANDO SECUENCIAS TEMPORALES")
        print("="*70)
        
        # Definir columnas de features
        base_features = ['Open', 'High', 'Low', 'Close']
        indicator_cols = [col for col in train_df.columns 
                         if col not in ['Date', 'ticker', 'Volume'] + base_features]
        
        feature_cols = base_features + indicator_cols
        print(f"Features usadas: {feature_cols}")
        
        feature_eng = FeatureEngineering(self.config)
        
        # Crear secuencias
        print("✓ Creando secuencias de train...")
        X_train, y_train, train_tickers, train_dates = feature_eng.create_sequences(
            train_df, feature_cols
        )
        
        print("✓ Creando secuencias de val...")
        X_val, y_val, val_tickers, val_dates = feature_eng.create_sequences(
            val_df, feature_cols
        )
        
        print("✓ Creando secuencias de test...")
        X_test, y_test, test_tickers, test_dates = feature_eng.create_sequences(
            test_df, feature_cols
        )
        
        print(f"\nShapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val:   {X_val.shape}")
        print(f"X_test:  {X_test.shape}")
        
        return (X_train, y_train, train_tickers, train_dates,
                X_val, y_val, val_tickers, val_dates,
                X_test, y_test, test_tickers, test_dates,
                feature_cols)
    
    def normalize_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Normaliza los datos"""
        print("\n" + "="*70)
        print("PASO 4: NORMALIZANDO DATOS")
        print("="*70)
        
        # Guardar forma original
        n_train, timesteps, n_features = X_train.shape
        
        # Reshape para normalizar
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # Fit scaler solo en train
        print("✓ Ajustando scaler en train...")
        X_train_scaled = self.feature_scaler.fit_transform(X_train_2d)
        X_val_scaled = self.feature_scaler.transform(X_val_2d)
        X_test_scaled = self.feature_scaler.transform(X_test_2d)
        
        # Reshape de vuelta
        X_train = X_train_scaled.reshape(X_train.shape)
        X_val = X_val_scaled.reshape(X_val.shape)
        X_test = X_test_scaled.reshape(X_test.shape)
        
        # Normalizar target
        y_train = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Guardar scalers
        scaler_path = self.config['paths']['models']
        with open(f"{scaler_path}/feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(f"{scaler_path}/target_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("✓ Scalers guardados")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Entrena un modelo específico"""
        print("\n" + "="*70)
        print(f"PASO 5: ENTRENANDO MODELO {model_name.upper()}")
        print("="*70)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model_path = f"{self.config['paths']['models']}/{model_name}_best.h5"
        
        # Seleccionar modelo
        if model_name == 'lstm':
            model = LSTMModel(self.config, input_shape)
        elif model_name == 'gru':
            model = GRUModel(self.config, input_shape)
        elif model_name == 'tcn':
            model = TCNModel(self.config, input_shape)
        elif model_name == 'tft':
            model = TFTModel(self.config, input_shape)
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")
        
        # Construir y mostrar arquitectura
        model.build()
        model.summary()
        
        # Entrenar
        history = model.train(X_train, y_train, X_val, y_val, model_path)
        
        # Guardar historial
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
        
        history_path = f"{self.config['paths']['models']}/{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"\n✓ Modelo {model_name} entrenado y guardado")
        print(f"✓ Mejor val_loss: {min(history.history['val_loss']):.6f}")
        
        return model, history
    
    def run(self, models=['lstm', 'gru', 'tcn', 'tft']):
        """Ejecuta el pipeline completo"""
        start_time = datetime.now()
        print("\n" + "="*70)
        print("INICIANDO PIPELINE DE ENTRENAMIENTO")
        print(f"Fecha: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Paso 1-4: Preparar datos
        train_df, val_df, test_df = self.load_and_prepare_data()
        train_df, val_df, test_df = self.engineer_features(train_df, val_df, test_df)
        
        results = self.create_sequences(train_df, val_df, test_df)
        (X_train, y_train, _, _, X_val, y_val, _, _, 
         X_test, y_test, _, _, feature_cols) = results
        
        X_train, y_train, X_val, y_val, X_test, y_test = self.normalize_data(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Guardar metadata
        metadata = {
            'feature_cols': feature_cols,
            'input_shape': list(X_train.shape[1:]),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'config': self.config
        }
        
        with open(f"{self.config['paths']['models']}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Paso 5: Entrenar modelos
        trained_models = {}
        for model_name in models:
            try:
                model, history = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                trained_models[model_name] = {
                    'model': model,
                    'history': history
                }
            except Exception as e:
                print(f"\n❌ ERROR entrenando {model_name}: {e}")
                continue
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO")
        print("="*70)
        print(f"Duración total: {duration}")
        print(f"Modelos entrenados: {list(trained_models.keys())}")
        print(f"Archivos guardados en: {self.config['paths']['models']}")
        print("="*70 + "\n")
        
        return trained_models

if __name__ == "__main__":
    # Entrenar todos los modelos
    pipeline = TrainingPipeline()
    
    # Puedes elegir qué modelos entrenar
    # models = ['lstm', 'gru']  # Solo estos dos
    models = ['lstm','gru','tcn', 'tft']  # Todos
    
    trained_models = pipeline.run(models=models)