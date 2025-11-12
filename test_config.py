# test_config.py
from src.utils import load_config, create_directories
from src.data_loader import StockDataLoader

# Cargar configuración
config = load_config()
print("✓ Config cargado correctamente")

# Crear directorios
create_directories(config)
print("✓ Directorios creados")

# Probar data loader
loader = StockDataLoader(config)
df = loader.load_full_data()

print(f"\n📊 Dataset filtrado:")
print(f"   Filas: {len(df):,}")
print(f"   Tickers: {df['ticker'].nunique()}")
print(f"   Rango: {df['Date'].min().date()} → {df['Date'].max().date()}")

# Hacer splits
train, val, test = loader.split_data(df)

print(f"\n✅ Splits creados:")
print(f"   Train: {len(train):,} filas")
print(f"   Val:   {len(val):,} filas")
print(f"   Test:  {len(test):,} filas")