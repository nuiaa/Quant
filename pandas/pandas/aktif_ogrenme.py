import sqlite3
import pandas as pd
import numpy as np

DB_FILE = 'telemetri_otopsi.sqlite'

def veritabanini_oku():
    print("[AKTİF ÖĞRENME] Hata Otopsi Veritabanı (telemetri_otopsi.sqlite) taranıyor...")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Sadece kapanmış işlemleri (BEKLEMEDE olmayan) getir
            df = pd.read_sql_query('''
                SELECT * FROM Karar_Anlari 
                WHERE Sonuc_Tipi != 'BEKLEMEDE'
            ''', conn)
            
            print(f"[AKTİF ÖĞRENME] Tamamlanmış {len(df)} adet işlem bulundu.")
            return df
    except Exception as e:
        print(f"[HATA] Veritabanı okunamadı: {e}")
        return pd.DataFrame()

def veriyi_hazirla(df):
    if df.empty:
        print("[AKTİF ÖĞRENME] Yeterli işlem kaydı yok, öğrenme iptal edildi.")
        return
        
    print("[AKTİF ÖĞRENME] Hatalar inceleniyor ve özellik matrisleri (X, Y) oluşturuluyor...")
    # Örnek kurgu:
    # Kar_Alindi (Kazanilan_Miktar > 0) -> Doğru Karar (1)
    # Stop_Oldu (Kazanilan_Miktar < 0) -> Hatalı Karar (0)
    
    # 20 Teknik ve 13 Makro sütunu X olarak ayrılacak
    # Bu veriler PyTorch modeline tekrar sokularak Transfer Learning (Fine-Tuning) yapılacak.
    
    # Şimdilik Taslak:
    kazananlar = len(df[df['Sonuc_Tipi'].str.contains('KAR_ALINDI', na=False)])
    kaybedenler = len(df[df['Sonuc_Tipi'].str.contains('STOP_OLDU', na=False)])
    
    print(f"Bilanço -> Kazanan Kararlar: {kazananlar}, Kaybeden Kararlar: {kaybedenler}")
    print("[AKTİF ÖĞRENME] Yapay zekanın kendi hatalarından öğrenme (Backpropagation) aşaması kodlanmaya hazır!")

if __name__ == "__main__":
    islem_gecmisi = veritabanini_oku()
    veriyi_hazirla(islem_gecmisi)
