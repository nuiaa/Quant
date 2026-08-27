import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import time
import sys
import io
import urllib.request
from datetime import datetime

# sys.stdout encoding reconfiguration for absolute safe terminal handling
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

VERITABANI_ADI = "yapay_zeka_veritabani.sqlite"
BASLANGIC_TARIHI = "2000-01-01"
BITIS_TARIHI = datetime.now().strftime("%Y-%m-%d")

def main():
    print("=" * 60)
    print("  S&P 500 COGRAFISI: TOPLU GERCEK VERI MADENCILIGI PANELI")
    print("=" * 60)
    
    # 1. ADIM: Wikipedia'dan S&P 500 Ticker Listesini Çek
    print("[1/4] Wikipedia üzerinden güncel S&P 500 şirket listesi çekiliyor...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        sp500_tablo = pd.read_html(io.StringIO(html_content))
        df_sp500 = sp500_tablo[0]
        tickers = df_sp500['Symbol'].tolist()
        print(f"  • Toplam {len(tickers)} adet şirket sembolü başarıyla çekildi.")
    except Exception as e:
        print(f"  • Hata: Şirket listesi Wikipedia'dan çekilemedi: {e}")
        return
    
    # 2. ADIM: Sembolleri Yahoo Finance ve SQLite Uyumluluğu İçin Düzenle
    print("\n[2/4] Semboller Yahoo Finance ve SQLite uyumluluğu için temizleniyor...")
    # Ticker mapping for downloading: dots replaced by dashes
    yf_tickers = [t.replace('.', '-') for t in tickers]
    # Reverse mapping from cleaned yf_ticker to final table name (dots and dashes to underscores)
    yf_to_table = {t.replace('.', '-'): t.replace('.', '_').replace('-', '_') for t in tickers}
    
    print("  • Temizleme tamamlandı. (Örn: BRK.B -> BRK-B (İndirme) -> BRK_B (Veritabanı))")

    # 3. ADIM: Yahoo Finance'den Çoklu İş Parçacıklı Toplu İndirme
    print(f"\n[3/4] 500 şirketin 26 yıllık günlük geçmiş verileri indiriliyor...")
    print(f"  • Başlangıç Tarihi : {BASLANGIC_TARIHI}")
    print(f"  • Bitiş Tarihi     : {BITIS_TARIHI}")
    print("  • İndirme yöntemi  : Eşzamanlı Çoklu İş Parçacığı (Multi-threading, 20 Threads)")
    
    t_basla = time.time()
    try:
        # yf.download is highly optimized for downloading multiple tickers concurrently
        data = yf.download(yf_tickers, start=BASLANGIC_TARIHI, end=BITIS_TARIHI, group_by='ticker', progress=True, threads=20)
        t_sure = time.time() - t_basla
        print(f"  • İndirme tamamlandı! (Süre: {t_sure:.2f} saniye)")
    except Exception as e:
        print(f"  • Hata: Veri toplu olarak indirilemedi: {e}")
        return

    # 4. ADIM: Verileri SQLite Veritabanına Yaz ve İndeksle
    print("\n[4/4] İndirilen veriler SQLite veritabanına bireysel tablolar halinde yazılıyor...")
    
    basarili_sayisi = 0
    toplam_satir = 0
    
    with sqlite3.connect(VERITABANI_ADI) as conn:
        for yf_ticker in yf_tickers:
            table_name = yf_to_table.get(yf_ticker, yf_ticker)
            try:
                # Extract columns for this ticker
                # If yf.download was successful, ticker will be in group_by columns
                if yf_ticker not in data.columns.levels[0]:
                    continue
                    
                df_ticker = data[yf_ticker].copy()
                
                # Drop completely empty/NaN rows
                df_ticker.dropna(subset=['Close'], inplace=True)
                
                if df_ticker.empty:
                    continue
                    
                # Keep only ham OHLCV
                ham_sutunlar = ['Open', 'High', 'Low', 'Close', 'Volume']
                df_ticker = df_ticker[[col for col in ham_sutunlar if col in df_ticker.columns]]
                
                # Format date index
                df_ticker.index = df_ticker.index.strftime('%Y-%m-%d')
                df_ticker.index.name = 'Datetime'
                
                # Save to sqlite
                df_ticker.to_sql(table_name, conn, if_exists='replace', index=True)
                conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_datetime ON {table_name} (Datetime)")
                
                basarili_sayisi += 1
                toplam_satir += len(df_ticker)
                
                if basarili_sayisi % 50 == 0:
                    print(f"  • [{basarili_sayisi}/{len(yf_tickers)}] şirket SQLite'a başarıyla yazıldı...")
                    
            except Exception as e:
                # Skip silently on individual ticker errors to prevent spam
                pass
                
    print("\n" + "=" * 60)
    print("  S&P 500 VERI MADENCILIGI TAMAMLANDI!")
    print("=" * 60)
    print(f"  • Başarıyla Yazılan Şirket Sayısı : {basarili_sayisi}/{len(yf_tickers)}")
    print(f"  • Veritabanına Eklenen Satır Sayısı: {toplam_satir:,} satır")
    print(f"  • Hedef Veritabanı Dosyası         : {VERITABANI_ADI}")
    print("=" * 60)

if __name__ == "__main__":
    main()
