import pandas as pd
import yaml
from tqdm import tqdm

print("Cargando datos...")

# Cargar config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Cargar datos
df = pd.read_csv('data/processed/stocks_full.csv', parse_dates=['Date'])

print(f"Dataset: {len(df):,} filas, {df['ticker'].nunique()} tickers")

# Aplicar filtros del config
start_date = pd.to_datetime(config['data']['start_date'])
train_end = pd.to_datetime(config['data']['train_end'])
val_end = pd.to_datetime(config['data']['val_end'])

df = df[df['Date'] >= start_date]
print(f"Después de filtrar desde {start_date.date()}: {len(df):,} filas, {df['ticker'].nunique()} tickers")

# PRE-FILTRAR RÁPIDAMENTE
print("\nPre-filtrando tickers...")

# Agrupar una sola vez (MUCHO más rápido)
grouped = df.groupby('ticker').agg({
    'Date': ['min', 'max', 'count'],
    'Close': 'mean'
}).reset_index()

grouped.columns = ['ticker', 'date_min', 'date_max', 'days', 'avg_price']
grouped['years'] = (grouped['date_max'] - grouped['date_min']).dt.days / 365

# Filtrar por años de cobertura
min_years = config['data']['min_years_coverage']
grouped = grouped[grouped['years'] >= min_years]
print(f"Tickers con >= {min_years} años: {len(grouped)}")

# Filtrar por precio ($50-$1000)
grouped = grouped[(grouped['avg_price'] >= 100) & (grouped['avg_price'] <= 500)]
print(f"Tickers en rango $100-$500: {len(grouped)}")

# Verificar presencia en splits (más rápido)
print("\nVerificando cobertura en train/val/test...")
valid_tickers = []

for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Verificando"):
    ticker = row['ticker']
    ticker_df = df[df['ticker'] == ticker]
    
    # Verificar que tenga datos en train/val/test
    has_train = len(ticker_df[ticker_df['Date'] <= train_end]) >= 30
    has_val = len(ticker_df[(ticker_df['Date'] > train_end) & (ticker_df['Date'] <= val_end)]) >= 30
    has_test = len(ticker_df[ticker_df['Date'] > val_end]) >= 30
    
    if has_train and has_val and has_test:
        valid_tickers.append({
            'ticker': ticker,
            'years': row['years'],
            'avg_price': row['avg_price'],
            'days': row['days']
        })

# Ordenar por años de datos
valid_tickers = sorted(valid_tickers, key=lambda x: x['years'], reverse=True)

print("\n" + "="*70)
print(f"✓ ENCONTRADOS {len(valid_tickers)} TICKERS VÁLIDOS")
print("="*70)

if len(valid_tickers) == 0:
    print("\n⚠️  NO SE ENCONTRARON TICKERS VÁLIDOS")
    print("   Prueba relajar los filtros en config.yaml:")
    print("   - start_date: '2019-01-01' (o más reciente)")
    print("   - min_years_coverage: 3 (en vez de 5)")
    exit()

# Mostrar top 50
n_show = min(50, len(valid_tickers))
print(f"\nTop {n_show} tickers por años de cobertura:\n")
for i, t in enumerate(valid_tickers[:n_show], 1):
    print(f"{i:2d}. {t['ticker']:6s} - {t['years']:4.1f} años - ${t['avg_price']:6.2f} avg - {t['days']:4d} días")

# Mostrar por rangos de precio
print("\n" + "="*70)
print("DISTRIBUCIÓN POR RANGO DE PRECIO")
print("="*70)

ranges = [
    (50, 100, "Bajo"),
    (100, 200, "Medio-Bajo"),
    (200, 400, "Medio"),
    (400, 1000, "Alto")
]

for min_p, max_p, label in ranges:
    in_range = [t for t in valid_tickers if min_p <= t['avg_price'] < max_p]
    print(f"\n{label} (${min_p}-${max_p}): {len(in_range)} tickers")
    if in_range:
        tickers_str = ', '.join([t['ticker'] for t in in_range[:20]])
        if len(in_range) > 20:
            tickers_str += f" ... (+{len(in_range)-20} más)"
        print(f"  {tickers_str}")

# Generar config con top 50
n_config = min(len(valid_tickers), len(valid_tickers))
print("\n" + "="*70)
print(f"CONFIGURACIÓN RECOMENDADA (TOP {n_config})")
print("="*70)
print("\nselected_tickers:")
for t in valid_tickers[:n_config]:
    print(f"  - \"{t['ticker']}\"  # {t['years']:.1f}y - ${t['avg_price']:.2f}")

# Guardar
output_file = f'data/processed/top_{n_config}_tickers.txt'
with open(output_file, 'w') as f:
    f.write('\n'.join([t['ticker'] for t in valid_tickers[:n_config]]))

print(f"\n✓ Lista guardada en: {output_file}")
print(f"✓ Total de tickers válidos disponibles: {len(valid_tickers)}")