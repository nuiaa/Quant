import sqlite3
import os
import uuid
from datetime import datetime

DB_FILE = 'telemetri_otopsi.sqlite'

def db_kurulum_yap():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Karar_Anlari (
                Islem_ID TEXT PRIMARY KEY,
                Zaman TEXT,
                Sembol TEXT,
                AI_Olasilik_Long REAL,
                AI_Olasilik_Short REAL,
                Alinan_Aksiyon TEXT,
                Anlik_Fiyat REAL,
                
                -- OTOPSİ SONUÇLARI --
                Sonuc_Tipi TEXT,
                Gerceklesme_Zamani TEXT,
                Kazanilan_Miktar REAL,
                Max_Drawdown REAL,
                
                -- TEKNİK ÖZELLİKLER (20) --
                Fiyat_EMA20_Farki REAL, Fiyat_EMA200_Farki REAL, RSI REAL, 
                MACD REAL, MACD_Histogram REAL, ATR_Yuzde REAL, BB_Pozisyon REAL, 
                Ust_Fitil_Gucu REAL, Alt_Fitil_Gucu REAL, Hacim_Patlamasi_Orani REAL, 
                Fiyat_Degisimi_5G REAL, RSI_Degisimi_5G REAL, Uyumsuzluk_Skoru REAL, 
                Likidite_Avi_Siddeti REAL, BB_Sikisma_Orani REAL, HA_Govde_Gucu REAL, 
                VWAP_Uzaklik REAL, VWAP_Egilim_5G REAL, PCR_Seviyesi REAL, Net_Opsiyon_Gucu REAL,
                
                -- MAKRO ÖZELLİKLER (13) --
                VIX REAL, DXY REAL, RS_Teknoloji REAL, RS_Finans REAL, RS_Saglik REAL, 
                RS_Enerji REAL, RS_Sanayi REAL, RS_Defansif REAL, RS_Hammadde REAL,
                Gun_Sin REAL, Gun_Cos REAL, Ay_Sin REAL, Ay_Cos REAL
            )
        ''')
        conn.commit()

# İlk yüklendiğinde tabloları kontrol et
db_kurulum_yap()

def canli_durumu_kara_kutuya_yaz(
    sembol, ai_long, ai_short, aksiyon, fiyat, 
    teknik_dict, makro_dict
):
    """
    Yapay zekanın karar anındaki tüm anatomisini (33 kurşun)
    hafta sonu analizi (Active Learning) için SQLite veritabanına kaydeder.
    Geriye, pozisyon kapandığında eşleştirmek üzere bir Islem_ID döner.
    """
    islem_id = str(uuid.uuid4())
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Null veya NaN gelme ihtimaline karşı varsayılan 0.0 atıyoruz
    def get_val(d, k):
        val = d.get(k, 0.0)
        import pandas as pd
        if pd.isna(val): return 0.0
        return float(val)
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Karar_Anlari (
                    Islem_ID, Zaman, Sembol, AI_Olasilik_Long, AI_Olasilik_Short, Alinan_Aksiyon, Anlik_Fiyat,
                    Sonuc_Tipi, Gerceklesme_Zamani, Kazanilan_Miktar, Max_Drawdown,
                    Fiyat_EMA20_Farki, Fiyat_EMA200_Farki, RSI, MACD, MACD_Histogram, ATR_Yuzde, BB_Pozisyon,
                    Ust_Fitil_Gucu, Alt_Fitil_Gucu, Hacim_Patlamasi_Orani, Fiyat_Degisimi_5G, RSI_Degisimi_5G,
                    Uyumsuzluk_Skoru, Likidite_Avi_Siddeti, BB_Sikisma_Orani, HA_Govde_Gucu, VWAP_Uzaklik,
                    VWAP_Egilim_5G, PCR_Seviyesi, Net_Opsiyon_Gucu,
                    VIX, DXY, RS_Teknoloji, RS_Finans, RS_Saglik, RS_Enerji, RS_Sanayi, RS_Defansif, RS_Hammadde,
                    Gun_Sin, Gun_Cos, Ay_Sin, Ay_Cos
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
            ''', (
                islem_id, zaman, sembol, float(ai_long), float(ai_short), aksiyon, float(fiyat),
                'BEKLEMEDE', None, 0.0, 0.0,
                get_val(teknik_dict, 'Fiyat_EMA20_Farki'), get_val(teknik_dict, 'Fiyat_EMA200_Farki'), get_val(teknik_dict, 'RSI'), 
                get_val(teknik_dict, 'MACD'), get_val(teknik_dict, 'MACD_Histogram'), get_val(teknik_dict, 'ATR_Yuzde'), get_val(teknik_dict, 'BB_Pozisyon'), 
                get_val(teknik_dict, 'Ust_Fitil_Gucu'), get_val(teknik_dict, 'Alt_Fitil_Gucu'), get_val(teknik_dict, 'Hacim_Patlamasi_Orani'), 
                get_val(teknik_dict, 'Fiyat_Degisimi_5G'), get_val(teknik_dict, 'RSI_Degisimi_5G'), get_val(teknik_dict, 'Uyumsuzluk_Skoru'), 
                get_val(teknik_dict, 'Likidite_Avi_Siddeti'), get_val(teknik_dict, 'BB_Sikisma_Orani'), get_val(teknik_dict, 'HA_Govde_Gucu'), 
                get_val(teknik_dict, 'VWAP_Uzaklik'), get_val(teknik_dict, 'VWAP_Egilim_5G'), get_val(teknik_dict, 'PCR_Seviyesi'), get_val(teknik_dict, 'Net_Opsiyon_Gucu'),
                
                get_val(makro_dict, 'VIX'), get_val(makro_dict, 'DXY'), get_val(makro_dict, 'RS_Teknoloji'), get_val(makro_dict, 'RS_Finans'), get_val(makro_dict, 'RS_Saglik'), 
                get_val(makro_dict, 'RS_Enerji'), get_val(makro_dict, 'RS_Sanayi'), get_val(makro_dict, 'RS_Defansif'), get_val(makro_dict, 'RS_Hammadde'),
                get_val(makro_dict, 'Gun_Sin'), get_val(makro_dict, 'Gun_Cos'), get_val(makro_dict, 'Ay_Sin'), get_val(makro_dict, 'Ay_Cos')
            ))
            conn.commit()
            return islem_id
    except Exception as e:
        print(f"[TELEMETRİ HATASI] Karar anı kayıt edilemedi: {e}")
        return None

def otopsi_sonucunu_guncelle(islem_id, sonuc_tipi, pnl, max_dd=0.0):
    """
    Pozisyon kapandığında (Kâr, Zarar, AI Kaçış vs.) ilgili kaydın
    sonuç kısımlarını günceller.
    sonuc_tipi: 'KAR_ALINDI', 'STOP_OLDU', 'ZAMAN_ASIMI', 'AI_KACIS' vb.
    """
    if not islem_id:
        return
        
    kapanis_zamani = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE Karar_Anlari 
                SET Sonuc_Tipi = ?, Gerceklesme_Zamani = ?, Kazanilan_Miktar = ?, Max_Drawdown = ?
                WHERE Islem_ID = ?
            ''', (sonuc_tipi, kapanis_zamani, float(pnl), float(max_dd), islem_id))
            conn.commit()
    except Exception as e:
        print(f"[TELEMETRİ HATASI] Otopsi sonucu güncellenemedi: {e}")
