import yfinance as yf
import sqlite3
import pandas as pd

DB_NAME = "yapay_zeka_veritabani.sqlite"

print("SPY verisi indiriliyor...")
df = yf.download("SPY", start="1990-01-01", progress=False)

if not df.empty:
    print(f"{len(df)} satir veri indirildi.")
    
    # Check if there is a multi-index column from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
        
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    
    print("Veritabanina kaydediliyor...")
    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql("SPY", conn, if_exists='replace', index=True)
    print("SPY verisi basariyla veritabanina eklendi!")
else:
    print("SPY verisi indirilemedi!")
