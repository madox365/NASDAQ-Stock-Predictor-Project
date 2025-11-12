import pandas as pd

test_df = pd.read_csv('data/processed/test.csv')

# Ver TODOS los tickers y sus precios
ticker_prices = test_df.groupby('ticker')['Close'].agg(['mean', 'min', 'max', 'count'])
ticker_prices = ticker_prices.sort_values('mean', ascending=False)

print("="*70)
print("TODOS LOS TICKERS EN TU DATASET")
print("="*70)
print(ticker_prices)

print("\n" + "="*70)
print("TICKERS PROBLEMÁTICOS")
print("="*70)

# Tickers muy caros (>$500)
caros = ticker_prices[ticker_prices['mean'] > 500]
print(f"\n🔴 MUY CAROS (>$500) - ELIMINAR:")
print(caros)

# Tickers muy baratos (<$80)
baratos = ticker_prices[ticker_prices['mean'] < 80]
print(f"\n🔴 MUY BARATOS (<$80) - ELIMINAR:")
print(baratos)

# Tickers buenos ($80-$500)
buenos = ticker_prices[(ticker_prices['mean'] >= 80) & (ticker_prices['mean'] <= 500)]
print(f"\n✅ TICKERS BUENOS ($80-$500) - MANTENER:")
print(buenos)
print(f"\nTotal buenos: {len(buenos)}")