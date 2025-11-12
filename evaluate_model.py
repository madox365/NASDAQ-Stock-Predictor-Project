# evaluate_model.py
from src.train import TrainingPipeline
from src.evaluate import evaluate_from_pipeline

print("🔄 Preparando datos de test...")

# 1. Cargar y procesar datos (igual que en train.py)
pipeline = TrainingPipeline()

train_df, val_df, test_df = pipeline.load_and_prepare_data()
train_df, val_df, test_df = pipeline.engineer_features(train_df, val_df, test_df)

results = pipeline.create_sequences(train_df, val_df, test_df)
(X_train, y_train, _, _, X_val, y_val, _, _, 
 X_test, y_test, _, _, _) = results

X_train, y_train, X_val, y_val, X_test, y_test = pipeline.normalize_data(
    X_train, y_train, X_val, y_val, X_test, y_test
)

print(f"✅ Datos preparados: {len(X_test):,} muestras de test\n")

# 2. Evaluar modelo .h5
print("🧪 Evaluando modelos LSTM...")
evaluate_from_pipeline(X_test, y_test, models=['lstm', 'gru', 'tcn', 'tft'])