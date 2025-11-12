import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, calculate_metrics
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.tcn import TCNModel
from src.models.tft import TFTModel

class ModelEvaluator:
    def __init__(self, config_path='config.yaml'):
        """Inicializa el evaluador"""
        self.config = load_config(config_path)
        self.models_path = self.config['paths']['models']
        
        # Cargar scalers
        with open(f"{self.models_path}/feature_scaler.pkl", 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f"{self.models_path}/target_scaler.pkl", 'rb') as f:
            self.target_scaler = pickle.load(f)
        
        # Cargar metadata
        with open(f"{self.models_path}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.input_shape = tuple(self.metadata['input_shape'])
        
    def load_test_data(self):
        """Carga datos de test preprocesados"""
        print("\n" + "="*70)
        print("CARGANDO DATOS DE TEST")
        print("="*70)
        
        # Aquí deberías cargar los mismos datos que usaste en train.py
        # Por simplicidad, asumimos que ya tienes X_test, y_test guardados
        # En producción, deberías guardarlos en train.py
        
        # Por ahora, este método es un placeholder
        # En la práctica, ejecutarías el mismo pipeline que en train.py
        # hasta el punto de crear X_test, y_test
        
        print("⚠️  Ejecuta train.py primero para generar X_test, y_test")
        print("    O implementa la carga desde archivos CSV")
        
        return None, None
    
    def load_model(self, model_name):
        """Carga un modelo entrenado"""
        model_path = f"{self.models_path}/{model_name}_best.h5"
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo {model_name} no encontrado en {model_path}")
            return None
        
        # Crear instancia del modelo
        if model_name == 'lstm':
            model = LSTMModel(self.config, self.input_shape)
        elif model_name == 'gru':
            model = GRUModel(self.config, self.input_shape)
        elif model_name == 'tcn':
            model = TCNModel(self.config, self.input_shape)
        elif model_name == 'tft':
            model = TFTModel(self.config, self.input_shape)
        else:
            print(f"❌ Modelo desconocido: {model_name}")
            return None
        
        # Cargar pesos
        model.load_weights(model_path)
        print(f"✓ Modelo {model_name} cargado")
        
        return model
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evalúa un modelo en test set"""
        print(f"\nEvaluando {model_name.upper()}...")
        
        # Predicciones (ya normalizadas)
        y_pred_norm = model.predict(X_test)
        
        # Desnormalizar
        y_test_original = self.target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        y_pred_original = self.target_scaler.inverse_transform(
            y_pred_norm.reshape(-1, 1)
        ).flatten()
        
        # Calcular métricas
        metrics = calculate_metrics(y_test_original, y_pred_original)
        
        print(f"  RMSE: ${metrics['RMSE']:.4f}")
        print(f"  MAE:  ${metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.4f}")
        
        return {
            'model_name': model_name,
            'metrics': metrics,
            'y_true': y_test_original,
            'y_pred': y_pred_original
        }
    
    def plot_comparison(self, results):
        """Genera gráficos de comparación"""
        print("\n" + "="*70)
        print("GENERANDO VISUALIZACIONES")
        print("="*70)
        
        plots_path = self.config['paths']['plots']
        
        # 1. Comparación de métricas
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparación de Modelos', fontsize=16, fontweight='bold')
        
        models = [r['model_name'] for r in results]
        metrics_names = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics_names):
            ax = axes[idx // 2, idx % 2]
            values = [r['metrics'][metric] for r in results]
            
            bars = ax.bar(models, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(axis='y', alpha=0.3)
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{plots_path}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {plots_path}/metrics_comparison.png")
        plt.close()
        
        # 2. Predicciones vs Real (sample)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predicciones vs Valores Reales (primeras 500 muestras)', 
                     fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(results):
            ax = axes[idx // 2, idx % 2]
            
            sample_size = min(500, len(result['y_true']))
            x = range(sample_size)
            
            ax.plot(x, result['y_true'][:sample_size], 
                   label='Real', alpha=0.7, linewidth=2)
            ax.plot(x, result['y_pred'][:sample_size], 
                   label='Predicción', alpha=0.7, linewidth=2)
            
            ax.set_title(f"{result['model_name'].upper()}", fontweight='bold')
            ax.set_xlabel('Muestra')
            ax.set_ylabel('Precio ($)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plots_path}/predictions_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {plots_path}/predictions_comparison.png")
        plt.close()
        
        # 3. Scatter plots (predicción vs real)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predicción vs Real (Scatter Plot)', 
                     fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(results):
            ax = axes[idx // 2, idx % 2]
            
            ax.scatter(result['y_true'], result['y_pred'], 
                      alpha=0.3, s=10)
            
            # Línea diagonal perfecta
            min_val = min(result['y_true'].min(), result['y_pred'].min())
            max_val = max(result['y_true'].max(), result['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Predicción perfecta')
            
            ax.set_title(f"{result['model_name'].upper()} (R²={result['metrics']['R2']:.4f})",
                        fontweight='bold')
            ax.set_xlabel('Valor Real ($)')
            ax.set_ylabel('Predicción ($)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plots_path}/scatter_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {plots_path}/scatter_comparison.png")
        plt.close()
        
    def save_results_table(self, results):
        """Guarda tabla de resultados"""
        tables_path = self.config['paths']['tables']
        
        # Crear DataFrame
        data = []
        for result in results:
            row = {'Model': result['model_name'].upper()}
            row.update(result['metrics'])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Ordenar por RMSE (menor es mejor)
        df = df.sort_values('RMSE')
        
        # Guardar
        csv_path = f"{tables_path}/evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Tabla guardada: {csv_path}")
        
        # Mostrar en consola
        print("\n" + "="*70)
        print("RESULTADOS FINALES")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        return df
    
    def run(self, models=['lstm', 'gru', 'tcn', 'tft'], X_test=None, y_test=None):
        """Ejecuta evaluación completa"""
        print("\n" + "="*70)
        print("INICIANDO EVALUACIÓN DE MODELOS")
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        if X_test is None or y_test is None:
            print("\n⚠️  Debes proporcionar X_test y y_test")
            print("    Ejecuta train.py primero y pasa los datos aquí")
            return
        
        results = []
        
        for model_name in models:
            try:
                model = self.load_model(model_name)
                if model is None:
                    continue
                
                result = self.evaluate_model(model, model_name, X_test, y_test)
                results.append(result)
                
            except Exception as e:
                print(f"\n❌ ERROR evaluando {model_name}: {e}")
                continue
        
        if not results:
            print("\n❌ No se pudo evaluar ningún modelo")
            return
        
        # Generar visualizaciones y tablas
        self.plot_comparison(results)
        df_results = self.save_results_table(results)
        
        print("\n✅ Evaluación completada")
        
        return results, df_results

# Función auxiliar para usar con train.py
def evaluate_from_pipeline(X_test, y_test, models=['lstm', 'gru', 'tcn', 'tft']):
    """
    Función para llamar desde train.py
    
    Ejemplo:
        from src.evaluate import evaluate_from_pipeline
        evaluate_from_pipeline(X_test, y_test)
    """
    evaluator = ModelEvaluator()
    return evaluator.run(models=models, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    print("\n⚠️  Este script debe ejecutarse DESPUÉS de train.py")
    print("    O importa evaluate_from_pipeline() en train.py\n")