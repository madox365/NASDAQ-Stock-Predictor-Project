import yaml
import numpy as np
import random
import tensorflow as tf
import torch
import os

def load_config(config_path="config.yaml"):
    """Carga configuración desde YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    """Fija semillas para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_directories(config):
    """Crea directorios necesarios"""
    dirs = [
        config['paths']['results'],
        config['paths']['models'],
        config['paths']['plots'],
        config['paths']['tables'],
        config['data']['processed_path']
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de regresión"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }