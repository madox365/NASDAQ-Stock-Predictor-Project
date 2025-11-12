import pandas as pd
import yaml

# Cargar config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

selected_tickers = config['data']['selected_tickers']

print("="*70)
print(f"VERIFICANDO {len(selected_tickers)} TICKERS SELECCIONADOS")
print("="*70)

# Cargar datos completos
df = pd.read_csv('data/processed/stocks_full.csv', parse_dates=['Date'])

# Filtrar por fecha mínima
start_date = pd.to_datetime(config['data']['start_date'])
df = df[df['Date'] >= start_date]

# Verificar cada ticker
available = []
missing = []
insufficient_data = []

for ticker in selected_tickers:
    ticker_df = df[df['ticker'] == ticker]
    
    if len(ticker_df) == 0:
        missing.append(ticker)
    else:
        years = (ticker_df['Date'].max() - ticker_df['Date'].min()).days / 365
        
        if years >= config['data']['min_years_coverage']:
            available.append(ticker)
            print(f"✓ {ticker:6s} - {len(ticker_df):4d} días - {years:.1f} años")
        else:
            insufficient_data.append((ticker, years))
            print(f"✗ {ticker:6s} - {len(ticker_df):4d} días - {years:.1f} años (< {config['data']['min_years_coverage']})")

# Resumen
print("\n" + "="*70)
print("RESUMEN")
print("="*70)
print(f"✓ Disponibles: {len(available)} tickers")
print(f"✗ No encontrados: {len(missing)} tickers")
print(f"✗ Datos insuficientes: {len(insufficient_data)} tickers")

if missing:
    print(f"\nTickers NO ENCONTRADOS en el dataset:")
    print(f"  {', '.join(missing)}")

if insufficient_data:
    print(f"\nTickers con DATOS INSUFICIENTES:")
    for ticker, years in insufficient_data:
        print(f"  {ticker}: {years:.1f} años (necesita {config['data']['min_years_coverage']})")

# Verificar en splits finales
print("\n" + "="*70)
print("VERIFICANDO SPLITS FINALES")
print("="*70)

test_df = pd.read_csv('data/processed/test.csv')
test_tickers = test_df['ticker'].unique()

print(f"Tickers en test.csv: {len(test_tickers)}")
print(f"Seleccionados que llegaron a test: {sorted(test_tickers)}")

# Comparar
not_in_test = [t for t in available if t not in test_tickers]
if not_in_test:
    print(f"\n⚠️  Tickers disponibles pero NO en test:")
    print(f"   {', '.join(not_in_test)}")
    print(f"   (Probablemente no tienen datos en 2024-2025)")

# Generar config recomendado
print("\n" + "="*70)
print("RECOMENDACIÓN: USAR SOLO ESTOS TICKERS")
print("="*70)
print("\nCopia esto en tu config.yaml:\n")
print("selected_tickers:")
for ticker in sorted(test_tickers):
    print(f"  - \"{ticker}\"")

# Guardar para referencia
with open('data/processed/validated_tickers.txt', 'w') as f:
    f.write('\n'.join(sorted(test_tickers)))

print(f"\n✓ Lista guardada en: data/processed/validated_tickers.txt")