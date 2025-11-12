# preprocess_combine.py (CORREGIDO: ticker como string)
import pandas as pd
import os
from tqdm import tqdm

raw_folder = "data/raw/stocks"
output_file = "data/processed/stocks_full.csv"

if not os.path.exists(raw_folder):
    raise FileNotFoundError(f"No existe: {raw_folder}")

os.makedirs("data/processed", exist_ok=True)

csv_files = [f for f in os.listdir(raw_folder) 
             if f.lower().endswith('.csv') 
             and os.path.getsize(os.path.join(raw_folder, f)) > 100]

print(f"Archivos encontrados: {len(csv_files)}")

all_data = []
skipped = 0

for file in tqdm(csv_files, desc="Procesando"):
    ticker = os.path.splitext(file)[0].upper()  # "AAPL"
    path = os.path.join(raw_folder, file)
    
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        if df.empty or len(df.columns) < 6:
            skipped += 1
            continue

        # Verificar columnas
        cols = df.columns.str.lower().tolist()
        expected = ['ticker', 'date', 'open', 'high', 'low', 'close']
        if cols != expected:
            skipped += 1
            continue

        # Renombrar
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })

        # Convertir
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Volume'] = 0

        # ELIMINAR columna 'ticker' original y asignar la correcta
        if 'ticker' in df.columns:
            df = df.drop(columns=['ticker'])
        
        df['ticker'] = ticker  # ← ASIGNAR DESPUÉS

        df = df.dropna(subset=['Date', 'Close'])
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']]

        all_data.append(df)

    except Exception as e:
        skipped += 1
        print(f"ERROR {file}: {e}")

if not all_data:
    raise ValueError("Todos los archivos fueron saltados.")

full_df = pd.concat(all_data, ignore_index=True)
full_df = full_df.sort_values(['ticker', 'Date']).reset_index(drop=True)
full_df.to_csv(output_file, index=False)

print(f"\nÉXITO")
print(f"→ {output_file}")
print(f"→ Filas: {len(full_df):,}")
print(f"→ Tickers únicos: {full_df['ticker'].nunique()}")
print(f"→ Ejemplo: {full_df['ticker'].unique()[:5]}")