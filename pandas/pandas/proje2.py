import sys

# sys.stdout encoding reconfiguration for absolute safe terminal handling
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
import json
import os
import time
import random
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import csv
from datetime import datetime, timedelta
from telemetri_motoru import canli_durumu_kara_kutuya_yaz, otopsi_sonucunu_guncelle

# Simülasyon (Zaman Makinesi) modunda zamanı kontrol etmek için global değişken
SIMULASYON_ZAMANI = None

# Gereksiz Pandas uyarılarını gizleyelim
warnings.filterwarnings('ignore')

def piyasa_haritasini_yukle(dosya_yolu="piyasa_haritasi.json"):
    """
    İnsan okunabilir hiyerarşik JSON dosyasını botun mikrosaniyede (O(1)) bulabileceği
    'Sembol -> Özellikler' sözlüğüne dönüştürür.
    """
    bot_hafizasi = {}
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            ham_veri = json.load(f)
            
        sektorler = ham_veri.get("SEKTORLER", {})
        
        for sektor_adi, sektor_verisi in sektorler.items():
            s_etf = sektor_verisi.get("ETF", "")
            s_karakter = sektor_verisi.get("Karakter", "")
            s_makro = sektor_verisi.get("Makro_Hassasiyet", {"VIX": 0, "DXY": 0})
            
            for endustri_adi, endustri_verisi in sektor_verisi.get("Endustriler", {}).items():
                e_etf = endustri_verisi.get("ETF", "")
                hisseler = endustri_verisi.get("Hisseler", [])
                
                # Her hisse için devasa bir DNA profili oluşturup RAM'e yazıyoruz
                for hisse in hisseler:
                    bot_hafizasi[hisse] = {
                        "Sektor": sektor_adi,
                        "Sektor_ETF": s_etf,
                        "Endustri": endustri_adi,
                        "Endustri_ETF": e_etf,
                        "Karakter": s_karakter,
                        "Makro_Agirlik": s_makro
                    }
        print(f"[SİSTEM] {len(bot_hafizasi)} adet varlığın Sektör/Endüstri DNA'sı RAM'e yüklendi.")
        return bot_hafizasi
        
    except Exception as e:
        print(f"[UYARI] piyasa_haritasi.json okunamadı: {e}")
        return {}

# Bot başlarken bu sözlüğü global olarak RAM'e alır
VARLIK_DNA_DEPOSU = piyasa_haritasini_yukle()

import sqlite3

class SQLiteCache:
    """
    Yahoo Finance isteklerini önlemek için SQLite veritabanını kullanan caching katmanı.
    """
    def __init__(self, db_adi="yapay_zeka_veritabani.sqlite"):
        self.db_adi = db_adi
        self._db_hazirla()
        
    def _db_hazirla(self):
        """Metadata tablosunu hazırlar."""
        try:
            with sqlite3.connect(self.db_adi) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        symbol TEXT,
                        period TEXT,
                        interval TEXT,
                        last_updated REAL,
                        PRIMARY KEY (symbol, period, interval)
                    )
                """)
        except Exception as e:
            print(f"[UYARI] SQLiteCache veritabanı hazırlanamadı: {e}")
            
    def _is_fresh(self, symbol, period, interval):
        """Önbellekteki verinin güncelliğini kontrol eder."""
        limitler = {
            "1m": 60,       # 1 dakika
            "15m": 900,     # 15 dakika
            "1h": 3600,     # 1 saat
            "1d": 14400,    # 4 saat (günlük veriler seyrek değişir)
        }
        max_yas = limitler.get(interval, 900)
        
        try:
            with sqlite3.connect(self.db_adi) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT last_updated FROM cache_metadata WHERE symbol=? AND period=? AND interval=?",
                    (symbol, period, interval)
                )
                row = cur.fetchone()
                if row:
                    import time
                    yas = time.time() - row[0]
                    if yas < max_yas:
                        return True
        except Exception as e:
            print(f"[UYARI] Cache kontrolü sırasında hata: {e}")
        return False

    def veri_getir(self, symbol, period="1y", interval="1d", bulk_df=None):
        """
        Cache güncelse veritabanından, değilse yfinance'ten çeker ve veritabanını günceller.
        bulk_df verilmişse directly cache'i günceller ve veri döndürür.
        """
        table_name = f"cache_{symbol}_{interval}".replace("=", "_").replace("-", "_").replace("^", "_").replace(".", "_")
        
        if bulk_df is not None:
            if not bulk_df.empty:
                try:
                    if isinstance(bulk_df.columns, pd.MultiIndex):
                        bulk_df.columns = bulk_df.columns.droplevel(1)
                    bulk_df.columns = [col.strip().capitalize() for col in bulk_df.columns]
                    
                    ham_sutunlar = ['Open', 'High', 'Low', 'Close', 'Volume']
                    clean_df = bulk_df[[col for col in ham_sutunlar if col in bulk_df.columns]].copy()
                    
                    if not clean_df.empty:
                        clean_df.index = clean_df.index.strftime('%Y-%m-%d %H:%M:%S' if interval != '1d' else '%Y-%m-%d')
                        clean_df.index.name = 'Datetime'
                        
                        import time
                        with sqlite3.connect(self.db_adi) as conn:
                            clean_df.to_sql(table_name, conn, if_exists='replace', index=True)
                            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_datetime ON {table_name} (Datetime)")
                            conn.execute(
                                "INSERT OR REPLACE INTO cache_metadata (symbol, period, interval, last_updated) VALUES (?, ?, ?, ?)",
                                (symbol, period, interval, time.time())
                            )
                except Exception as e:
                    print(f"[UYARI] Bulk veri SQLite cache yazılamadı ({symbol}): {e}")
            
            if SIMULASYON_ZAMANI is not None and not bulk_df.empty:
                return bulk_df[bulk_df.index <= SIMULASYON_ZAMANI]
            return bulk_df

        if self._is_fresh(symbol, period, interval):
            try:
                with sqlite3.connect(self.db_adi) as conn:
                    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn, index_col='Datetime')
                    if not df.empty:
                        df.index = pd.to_datetime(df.index)
                        if SIMULASYON_ZAMANI is not None:
                            return df[df.index <= SIMULASYON_ZAMANI]
                        return df
            except Exception as e:
                print(f"[UYARI] SQLiteCache okuma hatası ({symbol}): {e}. yfinance'ten indiriliyor...")

        try:
            import yfinance as yf
            df_new = yf.download(symbol, period=period, interval=interval, progress=False)
            if df_new.empty:
                try:
                    with sqlite3.connect(self.db_adi) as conn:
                        df_old = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn, index_col='Datetime')
                        if not df_old.empty:
                            df_old.index = pd.to_datetime(df_old.index)
                            print(f"[UYARI] yfinance boş döndü! Fallback olarak eski cache verisi kullanılıyor ({symbol}).")
                            if SIMULASYON_ZAMANI is not None:
                                return df_old[df_old.index <= SIMULASYON_ZAMANI]
                            return df_old
                except:
                    pass
                return df_new
            
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.droplevel(1)
            df_new.columns = [col.strip().capitalize() for col in df_new.columns]
            
            ham_sutunlar = ['Open', 'High', 'Low', 'Close', 'Volume']
            clean_df = df_new[[col for col in ham_sutunlar if col in df_new.columns]].copy()
            
            clean_df.index = clean_df.index.strftime('%Y-%m-%d %H:%M:%S' if interval != '1d' else '%Y-%m-%d')
            clean_df.index.name = 'Datetime'
            
            import time
            with sqlite3.connect(self.db_adi) as conn:
                clean_df.to_sql(table_name, conn, if_exists='replace', index=True)
                conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_datetime ON {table_name} (Datetime)")
                conn.execute(
                    "INSERT OR REPLACE INTO cache_metadata (symbol, period, interval, last_updated) VALUES (?, ?, ?, ?)",
                    (symbol, period, interval, time.time())
                )
            
            clean_df.index = pd.to_datetime(clean_df.index)
            if SIMULASYON_ZAMANI is not None:
                return clean_df[clean_df.index <= SIMULASYON_ZAMANI]
            return clean_df
        except Exception as e:
            print(f"[HATA] Veri indirme ve cache yazma hatası ({symbol}): {e}")
            try:
                with sqlite3.connect(self.db_adi) as conn:
                    df_old = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn, index_col='Datetime')
                    if not df_old.empty:
                        df_old.index = pd.to_datetime(df_old.index)
                        if SIMULASYON_ZAMANI is not None:
                            return df_old[df_old.index <= SIMULASYON_ZAMANI]
                        return df_old
            except:
                pass
            return pd.DataFrame()

# Sektörel Para Akışı İçin Takip Edilecek ETF'ler
SEKTOR_ETFLERI = {
    'XLK': 'RS_Teknoloji',
    'XLF': 'RS_Finans',
    'XLV': 'RS_Saglik',
    'XLE': 'RS_Enerji',
    'XLI': 'RS_Sanayi',
    'XLP': 'RS_Defansif',
    'XLB': 'RS_Hammadde'
}

# Global önbellek nesnesi
veri_deposu = SQLiteCache()

# ==========================================
# YAPAY ZEKA MODEL ENTEGRASYONU (AHTAPOT BEYİN NAKLİ)
# ==========================================
try:
    import torch
    from beyin_mimarisi import DinamikHiyerarsikModel  # YENİ AHTAPOT MODEL
    from sicak_motor import CustomMinMaxScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    try:
        from beyin_mimarisi import DinamikHiyerarsikModel
        from sicak_motor import CustomMinMaxScaler
    except ImportError:
        print("[UYARI] beyin_mimarisi veya sicak_motor modülleri bulunamadı. Simülasyon modunda devam ediliyor.")
        DinamikHiyerarsikModel = None
        CustomMinMaxScaler = None

# GPU veya CPU tespiti
if TORCH_AVAILABLE:
    Cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    Cihaz = "CPU (Simulasyon Modu)"

# Modelleri bir kez yükle ve RAM'de tut
AI_MODELS = {}
AI_MODEL_NAME = None

try:
    if TORCH_AVAILABLE:
        import glob
        for model_yolu in glob.glob("*_long_hiyerarsik_beyin.pth"):
            sektor = model_yolu.replace("_long_hiyerarsik_beyin.pth", "")
            try:
                model_long = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)
                model_long.load_state_dict(torch.load(model_yolu, map_location=Cihaz))
                model_long.eval()
                
                short_yolu = model_yolu.replace("_long_", "_short_")
                model_short = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)
                if os.path.exists(short_yolu):
                    model_short.load_state_dict(torch.load(short_yolu, map_location=Cihaz))
                model_short.eval()
                
                AI_MODELS[sektor] = {
                    "LONG": model_long,
                    "SHORT": model_short
                }
                print(f"[SISTEM] {sektor} sektörü için Ahtapot Beyinler başarıyla yüklendi!")
            except Exception as e:
                print(f"[UYARI] {sektor} modeli yüklenemedi: {e}")
                
        AI_MODEL_NAME = "ROUTER" if AI_MODELS else "SIMULASYON"
        if not AI_MODELS:
            print("[SISTEM] Hiçbir Yapay Zeka Beyni tam yüklenemedi, MOCK SIMULASYON modunda çalışıyor!")
    else:
        AI_MODEL_NAME = "SIMULASYON"
        print("[SISTEM] PyTorch kurulu değil, simulasyon modunda çalışıyor.")
except Exception as e:
    print(f"[UYARI] PyTorch modelleri yuklenirken hata olustu: {e}")
    AI_MODEL_NAME = "SIMULASYON"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ==========================================
# YAPILANDIRMALAR VE AYARLAR
# ==========================================
try:
    from config import (
        TELEGRAM_BOT_TOKEN, 
        TELEGRAM_CHAT_ID
    )
except ImportError:
    print("[UYARI] config.py dosyasi bulunamadi! Lutfen kimlik bilgileri icin config.py dosyasini olusturun.")
    TELEGRAM_BOT_TOKEN = "BOT_TOKENINIZI_BURAYA_YAZIN"
    TELEGRAM_CHAT_ID = "CHAT_IDNIZI_BURAYA_YAZIN"

# Test modu aktif edilirse (True yapılırsa), AAPL/MSFT ile sınırlandırılmış hızlı taramalar yapılır ve mock verilerle test edilir.
TEST_MODU = False



# Piyasa Mekanikleri Ayarları (Komisyon ve Slippage)
KOMISYON_ORANI = 0.001  # %0.1
KAYMA_ORANI = 0.0005    # %0.05

ADAY_DOSYASI = "aday_havuzu.json"
PORTFOY_DOSYASI = "aktif_portfoy.json" # Default
GECMIS_DOSYASI = "islem_gecmisi.csv" # Default
KARA_LISTE_DOSYASI = "kara_liste.json" # Default

# ==========================================
# YARDIMCI FONKSİYONLAR (HAFIZA VE ZAMAN YÖNETİMİ)
# ==========================================
def get_us_eastern_time():
    """Sistem yerel saatinden bağımsız olarak ABD Doğu Saatini (EST/EDT) hesaplar."""
    # UTC saati al (Python 3.12+ uyumlu)
    from datetime import timedelta, timezone
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # ABD Yaz Saati Uygulaması (Daylight Saving Time - DST) tespiti:
    # Mart ayının ikinci Pazar gününden Kasım ayının ilk Pazar gününe kadar EDT (UTC-4),
    # yılın geri kalanında EST (UTC-5) geçerlidir.
    yil = utc_now.year
    # Mart'ın ikinci pazarı (saat 07:00 UTC)
    dst_basla = datetime(yil, 3, 8 + (6 - datetime(yil, 3, 8).weekday()) % 7, 7)
    # Kasım'ın ilk pazarı (saat 06:00 UTC)
    dst_bitis = datetime(yil, 11, 1 + (6 - datetime(yil, 11, 1).weekday()) % 7, 6)
    
    if dst_basla <= utc_now < dst_bitis:
        offset = -4  # EDT
    else:
        offset = -5  # EST
        
    return utc_now + timedelta(hours=offset)

def dosya_oku(dosya_adi):
    if os.path.exists(dosya_adi):
        try:
            with open(dosya_adi, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[UYARI] {dosya_adi} dosyası bozuk veya okunamadı: {e}. Yedek dosyadan (.bak) geri yüklenmeye çalışılıyor...")
            bak_adi = dosya_adi + ".bak"
            if os.path.exists(bak_adi):
                try:
                    with open(bak_adi, 'r', encoding='utf-8') as f:
                        veri = json.load(f)
                    # Bozulan ana dosyayı yedekten kurtar ve düzelt
                    dosya_yaz(dosya_adi, veri)
                    return veri
                except Exception as e2:
                    print(f"[CRITICAL ERROR] Yedek dosyası da okunamadı: {e2}")
    return {}

def dosya_yaz(dosya_adi, veri):
    try:
        # JSON serileştirmeden önce 'Veri' anahtarını (veya Pandas DataFrame/Series içeren başka anahtarları) temizle
        temiz_veri = {}
        for k, v in veri.items():
            if isinstance(v, dict):
                temiz_veri[k] = {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "Veri"}
            else:
                temiz_veri[k] = v
        
        # 1. Mevcut dosya varsa, bozunmaya karşı önce .bak olarak yedekle
        bak_adi = dosya_adi + ".bak"
        if os.path.exists(dosya_adi):
            try:
                if os.path.exists(bak_adi):
                    os.remove(bak_adi)
                os.rename(dosya_adi, bak_adi)
            except Exception as e:
                print(f"[UYARI] Yedek dosyası (.bak) oluşturulamadı: {e}")
                
        # 2. Önce geçici bir .tmp dosyasına yaz (Atomic Write)
        tmp_adi = dosya_adi + ".tmp"
        with open(tmp_adi, 'w', encoding='utf-8') as f:
            json.dump(temiz_veri, f, cls=NpEncoder, ensure_ascii=False, indent=4)
            
        # 3. Yazma başarılı bittiğinde, atomically değiştir (os.replace Windows'ta da üzerine yazar)
        os.replace(tmp_adi, dosya_adi)
    except Exception as e:
        print(f"[Hata] {dosya_adi} dosyasına yazılırken hata oluştu: {e}")

def islem_kaydet(sembol, giris_fiyati, cikis_fiyati, adet, sebep, yon="LONG", kar_zarar_usd=None, giris_nedenleri=""):
    """Kapanan işlemi hesaplar ve Excel (CSV) dosyasına kaydeder."""
    global GECMIS_DOSYASI
    dosya = GECMIS_DOSYASI
    dosya_bos_mu = not os.path.exists(dosya) or os.path.getsize(dosya) == 0
    
    if kar_zarar_usd is None:
        if yon == "LONG":
            kar_zarar_usd = (cikis_fiyati - giris_fiyati) * adet
        else:
            kar_zarar_usd = (giris_fiyati - cikis_fiyati) * adet
            
    kar_zarar_yuzde = (kar_zarar_usd / (giris_fiyati * adet)) * 100
    global SIMULASYON_ZAMANI
    if SIMULASYON_ZAMANI is not None:
        zaman = pd.to_datetime(SIMULASYON_ZAMANI).strftime("%Y-%m-%d %H:%M:%S")
    else:
        zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(dosya, mode='a', newline='', encoding='utf-8') as f:
            yazici = csv.writer(f)
            # Dosya ilk defa oluşuyorsa veya boşsa başlıkları ekle
            if dosya_bos_mu:
                yazici.writerow(['Tarih', 'Sembol', 'Yön', 'Giris Fiyati', 'Cikis Fiyati', 'Adet', 'P&L (USD)', 'P&L (%)', 'Kapanis Sebebi', 'Giris Nedenleri'])
            
            yazici.writerow([zaman, sembol, yon, round(giris_fiyati, 2), round(cikis_fiyati, 2), adet, round(kar_zarar_usd, 2), round(kar_zarar_yuzde, 2), sebep, giris_nedenleri])
    except Exception as e:
        print(f"[Hata] Islem gecmise yazilamadi: {e}")

def yon_izni_ve_reversion_filtresi(spot_fiyat, ema200, ema20, rsi_degeri, sapma_esigi=0.20):
    """
    KUSURSUZ FİLTRE: Trend Takibi + Teyitli Mean-Reversion
    Mega boğalarda ezilmemek için Momentum Kırılım (EMA20 ve RSI) teyidi arar.
    """
    uzaklik_orani = (spot_fiyat - ema200) / ema200 if ema200 > 0 else 0
    
    # Risk Çarpanı: Fiyat aşırı şiştiğinde trende devam etsek bile lotumuzu küçültür.
    izinler = {"LONG": False, "SHORT": False, "Rejim": "", "Risk_Carpani": 1.0}

    # ==========================================
    # 1. BÖLGE: FİYAT EMA200 ÜZERİNDE (BOĞA BÖLGESİ)
    # ==========================================
    if spot_fiyat > ema200:
        # ANA KURAL: Boğada LONG her zaman serbesttir! Trend senin dostundur.
        izinler["LONG"] = True 
        
        # AŞIRI ALIM (BALON) KONTROLÜ
        if uzaklik_orani >= sapma_esigi:
            # Fiyat şişkin, LONG girmeye devam et ama riski YARIYA İNDİR!
            izinler["Risk_Carpani"] = 0.5 
            
            # SHORT İZNİ İÇİN TEYİT KONTROLÜ (Momentum Kırıldı mı?)
            # Fiyat 20 günlük ortalamayı aşağı kırdıysa VEYA RSI 70'in (aşırı alımın) altına düştüyse
            if spot_fiyat < ema20 or rsi_degeri < 70:
                izinler["SHORT"] = True
                izinler["Rejim"] = f"AŞIRI ALIM (+%{uzaklik_orani*100:.1f}) | Momentum Kırıldı -> SHORT İzni AKTİF"
            else:
                izinler["Rejim"] = f"AŞIRI ALIM (+%{uzaklik_orani*100:.1f}) | Ralli Devam Ediyor -> SHORT VETO (Sadece %50 Riskli LONG)"
        else:
            izinler["Rejim"] = "NORMAL BOĞA | Sadece LONG Serbest"

    # ==========================================
    # 2. BÖLGE: FİYAT EMA200 ALTINDA (AYI BÖLGESİ)
    # ==========================================
    else:
        # ANA KURAL: Ayı piyasasında SHORT her zaman serbesttir!
        izinler["SHORT"] = True 
        
        # AŞIRI SATIM (ÇÖKÜŞ) KONTROLÜ
        if uzaklik_orani <= -sapma_esigi:
            # Fiyat çöktü, SHORT girmeye devam et ama riski YARIYA İNDİR!
            izinler["Risk_Carpani"] = 0.5
            
            # LONG İZNİ İÇİN TEYİT KONTROLÜ (Bıçak sekti mi?)
            # Fiyat 20 günlük ortalamayı yukarı kırdıysa VEYA RSI 30'un (aşırı satımın) üzerine çıktıysa
            if spot_fiyat > ema20 or rsi_degeri > 30:
                izinler["LONG"] = True
                izinler["Rejim"] = f"AŞIRI SATIM (%{uzaklik_orani*100:.1f}) | Dönüş Teyit Edildi -> LONG İzni AKTİF"
            else:
                izinler["Rejim"] = f"AŞIRI SATIM (%{uzaklik_orani*100:.1f}) | Şelale Sürüyor -> LONG VETO (Sadece %50 Riskli SHORT)"
        else:
            izinler["Rejim"] = "NORMAL AYI | Sadece SHORT Serbest"
            
    return izinler

def dinamik_kademeli_trailing_stop(pozisyon_yonu, giris_fiyati, anlik_fiyat, mevcut_stop, atr_degeri):
    """
    Kâr oranına (ATR cinsinden) göre stop mesafesini dinamik olarak daraltan Kuantum Vitesi.
    """
    # 1. Kârı Hesapla
    if pozisyon_yonu == "LONG":
        kar_miktari = anlik_fiyat - giris_fiyati
        kar_atr_cinsinden = kar_miktari / atr_degeri if atr_degeri > 0 else 0
    else: # SHORT
        kar_miktari = giris_fiyati - anlik_fiyat
        kar_atr_cinsinden = kar_miktari / atr_degeri if atr_degeri > 0 else 0
        
    # 2. Vites (Çarpan) Belirleme - NEFES ALAN YENİ MATEMATİK
    if kar_atr_cinsinden >= 1.5:
        carpan = 1.0  # HEDEF BÖLGESİ: Kâr 1.5'e ulaştığında çok sıkma, 1.0 ATR geriden takip et.
        rejim_vitesi = "HEDEF BÖLGESİ KİLİDİ (1.0 ATR)"
    elif kar_atr_cinsinden >= 1.0:
        carpan = 1.5  # NEFES PAYI: Fiyat 1.0 ATR kâra geçince, stopu giriş fiyatının sadece 0.5 altına çek.
        rejim_vitesi = "NEFES ALAN KÂR EVRESİ (1.5 ATR)"
    else:
        carpan = 1.5  # BAŞLANGIÇ: İlk stop mesafesi
        rejim_vitesi = "BAŞLANGIÇ (1.5 ATR)"
        
    # 3. Yeni Stop Noktasını Hesapla
    yeni_stop = mevcut_stop
    
    if pozisyon_yonu == "LONG":
        hesaplanan_yeni_stop = anlik_fiyat - (atr_degeri * carpan)
        # Stop sadece YUKARI taşınabilir
        if hesaplanan_yeni_stop > mevcut_stop:
            yeni_stop = hesaplanan_yeni_stop
            
    elif pozisyon_yonu == "SHORT":
        hesaplanan_yeni_stop = anlik_fiyat + (atr_degeri * carpan)
        # Short'ta stop sadece AŞAĞI taşınabilir
        if hesaplanan_yeni_stop < mevcut_stop:
            yeni_stop = hesaplanan_yeni_stop
            
    return round(yeni_stop, 2), carpan, rejim_vitesi

def dinamik_lot_hesapla(mevcut_kasa, risk_yuzdesi, anlik_fiyat, anlik_atr, baslangic_stop_carpani=2.5):
    """
    Kurumsal fonların kullandığı 'Volatility-Adjusted Position Sizing' algoritması.
    Kasanın sadece belirlenen %'sini ATR mesafesine göre riske eder.
    """
    riske_edilen_dolar = mevcut_kasa * (risk_yuzdesi / 100.0)
    hisse_basi_risk_dolar = anlik_atr * baslangic_stop_carpani
    
    if hisse_basi_risk_dolar <= 0:
        return 0 
        
    hesaplanan_lot = int(riske_edilen_dolar / hisse_basi_risk_dolar)
    
    alinacak_tutar = hesaplanan_lot * anlik_fiyat
    if alinacak_tutar > mevcut_kasa:
        hesaplanan_lot = int(mevcut_kasa / anlik_fiyat)
        
    return hesaplanan_lot

def master_ai_karar_motoru(prob_long, prob_short, spot_fiyat, ema200, adx_degeri, hisse_sektoru, makro_spy_trend, esik_long=55.0, hacim_anomalisi_var_mi=False):
    """
    4 Kuantum Çözümünü Aynı Anda Uygulayan Master Filtre + Zırh Delici Bypass:
    1. İndirilmiş Gerçekçi Long Eşiği (%62)
    2. ADX (Trend Gücü) Aşısı
    3. Teknoloji Hissesi Short Yasağı (Hard Veto)
    4. Göreli Baskınlık (Probability Spread)
    5. Bypass: ADX manipülasyonuna karşı Hacim Patlaması veya Yüksek Yapay Zeka Güveni (>%75)
    """
    fark_long_lehine = prob_long - prob_short
    fark_short_lehine = prob_short - prob_long
    boga_piyasasi = spot_fiyat > ema200
    
    # SPY Boğa trendindeyse (veya hissenin kendisi Boğa'daysa makro proxy olarak) ve hisse teknoloji ise
    is_tech = hisse_sektoru.upper() in ["TEKNOLOJİ", "YAZILIM", "YARI İLETKEN", "TEKNOLOJI", "BILISIM"]
    if makro_spy_trend == "BOGA" and is_tech:
        short_yasak = True
    else:
        short_yasak = False

    # LONG KARARI (Asimetrik Zeka + ADX + Bypass)
    if prob_long >= esik_long and fark_long_lehine >= 10.0:
        if boga_piyasasi and adx_degeri >= 25.0:
            return "LONG", "GÜÇLÜ TREND: ADX Onaylı Asimetrik LONG (Rokete Binildi)"
        elif boga_piyasasi and adx_degeri < 25.0:
            if prob_long >= 75.0 or hacim_anomalisi_var_mi:
                return "LONG", "BYPASS: Kurumsal manipülasyon aşıldı! (ADX düşük ama AI %75+ emin veya Hacim Patlaması var)"
            return "PAS", "VETO: AI Long diyor ama ADX zayıf (Yatay piyasa gürültüsü)"
        else:
            return "LONG", "DİPTEN DÖNÜŞ: Ayı piyasasında tepki alımı yakalandı."

    # SHORT KARARI (Katı İspat Zorunluluğu)
    if prob_short >= 75.0 and fark_short_lehine >= 35.0:
        if short_yasak:
            return "PAS", "VETO: Teknoloji hissesinde Boğa piyasasında SHORT açılamaz! İntihar engellendi."
        return "SHORT", "NET ÇÖKÜŞ: Yüksek güvenli ve geniş farkla onaylanmış SHORT."

    return "PAS", "KARARSIZ: AI olasılıkları tatmin edici spread yaratamadı."

def short_borrow_maliyeti_hesapla(islem_tutari_dolari, elde_tutulan_saat, yillik_borrow_rate=0.08):
    """
    SHORT işlemler için aracı kuruma ödenen gizli 'Hisse Kiralama' maliyetini hesaplar.
    Açığa satışı zor (Hard-to-borrow) hisselerde oran %20'lere çıkabilir.
    """
    # 1 günden kısa süren (gün içi) işlemler için genellikle 1 günlük minimum faiz kesilir
    elde_tutulan_gun = max(1.0, elde_tutulan_saat / 24.0)
    
    # Yıllık oranı günlüğe çevir
    gunluk_faiz_orani = yillik_borrow_rate / 365.0
    
    # Kiralama Bedeli P&L'den düşülecek net dolar maliyetidir
    kiralama_maliyeti_dolari = islem_tutari_dolari * gunluk_faiz_orani * elde_tutulan_gun
    
    return round(kiralama_maliyeti_dolari, 2)

HESAP_DOSYASI = "sanal_hesap.json"

def hesap_bilgisi_getir():
    """Sanal kasadaki bakiyeyi okur. Dosya yoksa 10.000$ ile başlatır."""
    global HESAP_DOSYASI
    if os.path.exists(HESAP_DOSYASI):
        try:
            with open(HESAP_DOSYASI, 'r', encoding='utf-8') as f:
                icerik = f.read().strip()
                if not icerik:
                    raise ValueError("Dosya bos")
                hesap = json.loads(icerik)
                
                # Migrations / Updates
                guncelleme = False
                bugun_str = datetime.now().strftime("%Y-%m-%d")
                hafta_str = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")

                if "Gun_Baslangici" not in hesap or hesap["Gun_Baslangici"] != bugun_str:
                    hesap["Gun_Baslangici"] = bugun_str
                    hesap["Gun_Baslangic_Bakiyesi"] = hesap.get("Guncel_Bakiye", 10000.0)
                    guncelleme = True
                
                if "Hafta_Baslangici" not in hesap or hesap["Hafta_Baslangici"] != hafta_str:
                    hesap["Hafta_Baslangici"] = hafta_str
                    hesap["Hafta_Baslangic_Bakiyesi"] = hesap.get("Guncel_Bakiye", 10000.0)
                    guncelleme = True
                    
                if "Zirve_Bakiye" not in hesap:
                    hesap["Zirve_Bakiye"] = hesap.get("Guncel_Bakiye", 10000.0)
                    guncelleme = True
                else:
                    if hesap["Guncel_Bakiye"] > hesap["Zirve_Bakiye"]:
                        hesap["Zirve_Bakiye"] = hesap["Guncel_Bakiye"]
                        guncelleme = True

                if "Devre_Kesici_Bitis" not in hesap:
                    hesap["Devre_Kesici_Bitis"] = "2000-01-01"
                    guncelleme = True
                    
                if "Sistem_Kilitli" not in hesap:
                    hesap["Sistem_Kilitli"] = False
                    guncelleme = True
                    
                if guncelleme:
                    hesap_kaydet(hesap)
                    
                return hesap
        except Exception as e:
            print(f"[KRİTİK UYARI] {HESAP_DOSYASI} okunamadı veya bozuk! Hata: {e}")
            bak_dosyasi = HESAP_DOSYASI + ".bak"
            if os.path.exists(bak_dosyasi):
                try:
                    with open(bak_dosyasi, 'r', encoding='utf-8') as f:
                        hesap = json.load(f)
                    print(f"[BİLGİ] Yedek dosyasından ({bak_dosyasi}) başarıyla kurtarıldı.")
                    return hesap
                except Exception as bak_e:
                    print(f"[HATA] Yedek dosya da okunamadı: {bak_e}")
            
            print(f"[UYARI] Yeni, temiz hesap başlatılıyor...")
            pass
    
    bugun_str = datetime.now().strftime("%Y-%m-%d")
    hafta_str = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
    
    baslangic = {
        "Baslangic_Bakiyesi": 10000.0, 
        "Guncel_Bakiye": 10000.0, 
        "Toplam_Kar_Zarar": 0.0,
        "Gun_Baslangici": bugun_str,
        "Gun_Baslangic_Bakiyesi": 10000.0,
        "Hafta_Baslangici": hafta_str,
        "Hafta_Baslangic_Bakiyesi": 10000.0,
        "Zirve_Bakiye": 10000.0,
        "Devre_Kesici_Bitis": "2000-01-01",
        "Sistem_Kilitli": False
    }
    hesap_kaydet(baslangic)
    return baslangic

def hesap_kaydet(hesap_verisi):
    global HESAP_DOSYASI
    tmp_dosya = HESAP_DOSYASI + ".tmp"
    bak_dosyasi = HESAP_DOSYASI + ".bak"
    try:
        with open(tmp_dosya, 'w', encoding='utf-8') as f:
            json.dump(hesap_verisi, f, cls=NpEncoder, ensure_ascii=False, indent=4)
        if os.path.exists(HESAP_DOSYASI):
            import shutil
            shutil.copyfile(HESAP_DOSYASI, bak_dosyasi)
        os.replace(tmp_dosya, HESAP_DOSYASI)
    except Exception as e:
        print(f"[HATA] Hesap dosyasi kaydedilemedi: {e}")

def satis_islemi_gerceklestir(sembol, giris_fiyati, cikis_fiyati, adet, sebep, yon="LONG"):
    """Pozisyon kapatıldığında kâr/zararı kasaya ekler/düşer ve CSV geçmişine yazar."""
    hesap = hesap_bilgisi_getir()
    
    # 1. Komisyon ve kayma düşülmüş net kâr/zarar hesaplama
    aktif_portfoy = dosya_oku(PORTFOY_DOSYASI)
    portfoy_verisi = aktif_portfoy.get(sembol, {})
    
    # Giris tarihini bul ve elde tutulan sureyi hesapla
    giris_tarihi_str = portfoy_verisi.get("Giris_Tarihi", datetime.now().strftime("%Y-%m-%d"))
    try:
        giris_tarihi = datetime.strptime(giris_tarihi_str, "%Y-%m-%d")
    except:
        giris_tarihi = datetime.now()
    gecen_saat = (datetime.now() - giris_tarihi).total_seconds() / 3600.0

    borrow_maliyeti = 0.0
    # 1. Komisyon ve kayma düşülmüş net kâr/zarar hesaplama
    if yon == "LONG":
        kar_zarar_usd = adet * (cikis_fiyati * (1 - KAYMA_ORANI - KOMISYON_ORANI) - giris_fiyati * (1 + KAYMA_ORANI + KOMISYON_ORANI))
    else: # SHORT
        kar_zarar_usd = adet * (giris_fiyati * (1 - KAYMA_ORANI - KOMISYON_ORANI) - cikis_fiyati * (1 + KAYMA_ORANI + KOMISYON_ORANI))
        borrow_maliyeti = short_borrow_maliyeti_hesapla(adet * giris_fiyati, gecen_saat, 0.08)
        kar_zarar_usd -= borrow_maliyeti
        
    # 2. Kasaya iade edilecek nakit miktarı
    # Giriş esnasında kasadan düşülen tutarı hesapla veya portföyden oku
    islem_maliyeti = portfoy_verisi.get("Islem_Maliyeti", None)
    
    if islem_maliyeti is None:
        # Fallback (Geriye dönük uyumluluk)
        islem_maliyeti = adet * giris_fiyati * (1 + KAYMA_ORANI + KOMISYON_ORANI)
        
    # --- YENİ EKLENEN KISIM: GİRİŞ NEDENLERİNİ PORTFÖYDEN ÇEK ---
    ai_olasilik = portfoy_verisi.get("Yapay_Zeka_Olasiligi", 0)
    onaylar = portfoy_verisi.get("Onaylar", [])
    onaylar_str = ", ".join(onaylar) if onaylar else "Standart Sinyal"
    giris_nedenleri = f"AI:%{ai_olasilik} | {onaylar_str}"
    # -------------------------------------------------------------

    # --- OTOPSİ SONUCUNU GÜNCELLE (ACTIVE LEARNING İÇİN) ---
    telemetri_id = portfoy_verisi.get("Telemetri_ID")
    if telemetri_id:
        otopsi_sonucunu_guncelle(telemetri_id, sebep, kar_zarar_usd)
    # -------------------------------------------------------------

    # Kasaya net bakiye iadesi: Giriş Maliyeti + Net Kar/Zarar
    iade_tutari = islem_maliyeti + kar_zarar_usd
    hesap["Guncel_Bakiye"] += float(iade_tutari)
    
    hesap["Toplam_Kar_Zarar"] = hesap["Guncel_Bakiye"] - hesap["Baslangic_Bakiyesi"]
    
    # --- YENİ EKLENEN KISIM: DEVRE KESİCİ VE ATH KORUMASI ---
    guncel_b = hesap["Guncel_Bakiye"]
    
    if guncel_b > hesap.get("Zirve_Bakiye", guncel_b):
        hesap["Zirve_Bakiye"] = guncel_b
        hesap["Sistem_Kilitli"] = False
    elif guncel_b < hesap.get("Zirve_Bakiye", guncel_b) * 0.88:
        hesap["Sistem_Kilitli"] = True
        
    # Günlük / Haftalık Kayıp Kontrolü
    gunluk_baslangic = hesap.get("Gun_Baslangic_Bakiyesi", guncel_b)
    haftalik_baslangic = hesap.get("Hafta_Baslangic_Bakiyesi", guncel_b)
    
    if guncel_b < gunluk_baslangic * 0.96: # %4 Günlük kayıp
        hesap["Devre_Kesici_Bitis"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
    if guncel_b < haftalik_baslangic * 0.92: # %8 Haftalık kayıp
        gun_farki = 7 - datetime.now().weekday()
        if gun_farki == 0: gun_farki = 7
        hesap["Devre_Kesici_Bitis"] = (datetime.now() + timedelta(days=gun_farki)).strftime("%Y-%m-%d")
    # -------------------------------------------------------------
    
    hesap_kaydet(hesap)
    islem_kaydet(sembol, giris_fiyati, cikis_fiyati, adet, sebep, yon, kar_zarar_usd, giris_nedenleri)
    
    # --- YENİ EKLENEN KISIM: KARA LİSTE (REVENGE TRADE KALKANI) ---
    if kar_zarar_usd < 0:
        bugun = datetime.now().strftime("%Y-%m-%d")
        kara_liste = dosya_oku(KARA_LISTE_DOSYASI) if os.path.exists(KARA_LISTE_DOSYASI) else {}
        kara_liste[sembol] = bugun
        dosya_yaz(KARA_LISTE_DOSYASI, kara_liste)
        print(f"[REVENGE TRADE KALKANI] {sembol} zararla kapandığı için gün boyunca kara listeye alındı.")
    # -------------------------------------------------------------
    
    # Telegram Bildirimi Gönder
    # Kâr/Zarar yüzdesi işlem maliyeti üzerinden hesaplanır
    kar_zarar_yuzde = (kar_zarar_usd / islem_maliyeti) * 100 if islem_maliyeti != 0 else 0.0
    durum = "[KAR]" if kar_zarar_usd >= 0 else "[ZARAR]"
    yon_str = "AÇIĞA SATIŞ (SHORT)" if yon == "SHORT" else "LONG"
    
    uyari_ek = ""
    if borrow_maliyeti > 0:
        uyari_ek += f"\n*Borrow Maliyeti:* ${borrow_maliyeti:.2f} düşüldü."
        
    if hesap.get("Sistem_Kilitli", False):
        uyari_ek += "\n*DİKKAT:* Zirveden %12 düşüş gerçekleşti! Sistem Kilitlendi (ATH Guard)."
    elif hesap.get("Devre_Kesici_Bitis", "2000-01-01") > datetime.now().strftime("%Y-%m-%d"):
        uyari_ek += f"\n*DEVRE KESİCİ AKTİF!* (Maksimum kayıp limiti aşıldı. Tekrar aktif olma: {hesap['Devre_Kesici_Bitis']})"
    
    mesaj = (
        f"{durum} *{sembol} {yon_str} POZİSYONU KAPATILDI ({sebep})*\n\n"
        f"*Giriş Fiyatı:* ${giris_fiyati:.2f} | *Çıkış Fiyatı:* ${cikis_fiyati:.2f}\n"
        f"*İşlem Adedi:* {adet} Adet\n\n"
        f"*Net P&L (Kâr/Zarar):* ${kar_zarar_usd:.2f} ({kar_zarar_yuzde:+.2f}%){uyari_ek}\n"
        f"*Giriş Nedenleri:* {giris_nedenleri}\n\n"
        f"*(%0.1 Komisyon ve %0.05 Kayma düşülmüştür)*\n"
        f"============================\n"
        f"*Kasa Bakiyesi:* ${hesap['Guncel_Bakiye']:.2f}\n"
        f"*Toplam Kâr/Zarar:* ${hesap['Toplam_Kar_Zarar']:.2f}\n"
        f"{uyari_ek}"
    )
    send_telegram_message(mesaj)

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == "BOT_TOKENINIZI_BURAYA_YAZIN" or TELEGRAM_CHAT_ID == "CHAT_IDNIZI_BURAYA_YAZIN":
        print("\n[UYARI] Telegram bildirimleri icin bot yapilandirmasi eksik!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("\n[BILDIRIM] Telegram raporu basariyla gonderildi!")
        else:
            print(f"\n[HATA] Telegram Hatasi: {response.text}")
    except Exception as e:
        print(f"\n[HATA] Telegram Baglanti Hatasi: {e}")

def telegram_fotograf_gonder(dosya_yolu, mesaj):
    """Cizilen grafigi Telegram uzerinden resimli mesaj olarak gonderir."""
    if TELEGRAM_BOT_TOKEN == "BOT_TOKENINIZI_BURAYA_YAZIN" or TELEGRAM_CHAT_ID == "CHAT_IDNIZI_BURAYA_YAZIN":
        print("\n[UYARI] Telegram bot yapilandirmasi eksik, fotograf gonderilemedi.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(dosya_yolu, 'rb') as foto:
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': mesaj, 'parse_mode': 'Markdown'}
            files = {'photo': foto}
            response = requests.post(url, data=payload, files=files, timeout=20)
            if response.status_code == 200:
                print("\n[BILDIRIM] Mum grafigi ve Telegram fotografli mesaj basariyla gonderildi!")
            else:
                print(f"\n[HATA] Telegram Fotograf Gonderme Hatasi: {response.text}")
    except Exception as e:
        print(f"\n[HATA] Telegram Fotograf Gonderme Hatasi: {e}")
    finally:
        # Gonderdikten sonra bilgisayarda yer kaplamamasi icin resmi sil
        if os.path.exists(dosya_yolu):
            try:
                os.remove(dosya_yolu)
            except Exception as e:
                print(f"[Uyari] Gecici grafik dosyasi silinemedi: {e}")

def grafik_ciz_ve_kaydet(df, sembol, islem_turu, fiyat, stop, hedef):
    """Hissenin son 100 günlük verisiyle profesyonel bir mum grafiği çizer."""
    # Sadece son 100 günü al (Grafik çok sıkışık olmasın diye)
    df_plot = df.tail(100).copy()
    
    # İşlem gününü işaretlemek için boş bir liste oluştur
    isaretler = [np.nan] * len(df_plot)
    isaretler[-1] = df_plot['Low'].iloc[-1] * 0.98 if islem_turu == "AL" else df_plot['High'].iloc[-1] * 1.02
    
    isaret_rengi = 'g' if islem_turu == "AL" else 'r'
    isaret_sekli = '^' if islem_turu == "AL" else 'v'

    # Grafiğe eklenecek katmanlar (EMA'lar, RSI ve İşlem İşaretçisi)
    ek_katmanlar = [
        mpf.make_addplot(df_plot['EMA_5'], color='blue', width=1.5, title="EMA 5/20/200"),
        mpf.make_addplot(df_plot['EMA_20'], color='orange', width=1.5),
        mpf.make_addplot(df_plot['EMA_200'], color='black', width=2),
        mpf.make_addplot(df_plot['RSI'], panel=1, color='purple', ylabel='RSI (14)'),
        mpf.make_addplot(isaretler, type='scatter', markersize=200, marker=isaret_sekli, color=isaret_rengi)
    ]

    # Hedef ve Stop çizgileri (Sadece AL işleminde gösterilir)
    yatay_cizgiler = dict(hlines=[stop, hedef], colors=['r', 'g'], linestyle='-.') if islem_turu == "AL" else None

    # Çizim ayarları
    stil = mpf.make_mpf_style(base_mpf_style='yahoo', gridstyle=':')
    dosya_adi = f"{sembol}_{islem_turu}_Sinyali.png"
    
    baslik = f"{sembol} - KUSURSUZ {islem_turu} SINYALI (Fiyat: {fiyat})"
    
    # Grafiği çiz ve kaydet
    mpf.plot(df_plot, type='candle', style=stil, addplot=ek_katmanlar,
             hlines=yatay_cizgiler, volume=False, figratio=(16,9),
             figscale=1.2, title=baslik, savefig=dosya_adi)
    
    return dosya_adi


def on_bes_dakikalik_gecmis_kontrol(sembol, stop, hedef, yon="LONG"):
    """
    Botun uyuduğu son 15-20 dakikayı, 1 dakikalık mumlarla KRONOLOJİK olarak tarar.
    Önce hedefe mi yoksa stopa mı değdiğini kesin olarak tespit eder.
    """
    try:
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=30)
        
        # Son 30 dakikalık 1m verisini indirerek yfinance rate limitlerini ve veri hacmini minimize ediyoruz
        df_1m = yf.download(sembol, start=start_time, end=end_time, interval="1m", progress=False)
        if df_1m.empty:
            return "BEKLE"

        if isinstance(df_1m.columns, pd.MultiIndex):
            df_1m.columns = df_1m.columns.droplevel(1)
            
        # Son 30 dakikalık verinin tamamını (ya da son 20 mumunu) kontrol edelim
        son_20_dk = df_1m.tail(20)

        # Zaman sırasına (eskiden yeniye) göre her 1 dakikayı tek tek kontrol et
        for zaman, mum in son_20_dk.iterrows():
            if yon == "LONG":
                # Kârda iğne yeterlidir (Fiyat hedefe değmişse kârı cebe atar)
                if mum['High'] >= hedef:
                    return "HEDEF_VURULDU"
                    
                # Stop için sadece Low değil, CLOSE fiyatının stop'u kırmasını bekle
                if mum['Close'] <= stop:
                    return "STOP_VURULDU"
                    
            else: # SHORT
                if mum['Low'] <= hedef:
                    return "HEDEF_VURULDU"
                    
                if mum['Close'] >= stop:
                    return "STOP_VURULDU"
                
        return "BEKLE" # Hiçbirine değmemiş veya iğne atıp kapatmamış, işleme devam.
    except Exception as e:
        print(f"[{sembol}] 1m Mikro tarama yapilamadi: {e}")
        return "BEKLE"

def yapay_zeka_ile_analiz_yap(sembol, ad=""):
    """Canli veriyi ceker, önbelleğe bakar, tensore cevirir ve Master Long & Short Hibrit Modellere firlatarak olasiliklari alir."""
    if not AI_MODELS:
        return None
        
    try:
        from sicak_motor import veriyi_oku_ve_ozellikleri_hesapla
        
        # Son 5 gunun 15 dakikalik verisini önbelleğe indirmek ve tazelemek için veri_deposu.veri_getir çağırıyoruz
        veri_deposu.veri_getir(sembol, period="5d", interval="15m")
        
        table_name = f"cache_{sembol}_15m".replace("=", "_").replace("-", "_").replace("^", "_").replace(".", "_")
        
        # sicak_motor.py'deki veriyi_oku_ve_ozellikleri_hesapla fonksiyonu ile indikatörleri ve makro (VIX, DXY) verilerini dinamik hesapla
        # Canlı izlemede Hacim Çubuklarını (Volume Bars) etkinleştir
        df = veriyi_oku_ve_ozellikleri_hesapla(sembol, tablo_adi=table_name, kesme_zamani=SIMULASYON_ZAMANI, canli_mod=True)
        if df.empty or len(df) < 61: 
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # Repainting Önleme: Model girdileri için en son açık mumu (df.iloc[-1]) hariç tutup kapalı mumları alıyoruz
        son_60_mum = df.iloc[:-1].tail(60)
        
        makro_ozellikler = [
            'VIX', 'DXY', 
            'RS_Teknoloji', 'RS_Finans', 'RS_Saglik', 
            'RS_Enerji', 'RS_Sanayi', 'RS_Defansif', 'RS_Hammadde',
            'Gun_Sin', 'Gun_Cos', 'Ay_Sin', 'Ay_Cos'
        ]
        teknik_ozellikler = [
            'Fiyat_EMA20_Farki', 'Fiyat_EMA200_Farki', 'RSI', 
            'MACD', 'MACD_Histogram', 'ATR_Yuzde', 'BB_Pozisyon', 
            'Ust_Fitil_Gucu', 'Alt_Fitil_Gucu',       
            'Hacim_Patlamasi_Orani',                  
            'Fiyat_Degisimi_5G', 'RSI_Degisimi_5G',
            'Uyumsuzluk_Skoru', 'Likidite_Avi_Siddeti', 'BB_Sikisma_Orani',
            'HA_Govde_Gucu', 'VWAP_Uzaklik', 'VWAP_Egilim_5G',
            'PCR_Seviyesi', 'Net_Opsiyon_Gucu',
            'Fib_382_K_Uzaklik', 'Fib_618_K_Uzaklik',
            'Fib_382_U_Uzaklik', 'Fib_618_U_Uzaklik',
            'Alt_Golge_Orani', 'Ust_Golge_Orani', 'Hacim_Anomalisi',
            'Ayi_Uyumsuzlugu', 'Boga_Uyumsuzlugu', 'Frac_Close'
        ]
        
        # 1. HAM VERİLERİ İKİ KOLA AYIR
        # Teknik kol: Son 60 mumluk serüven (Matris)
        X_teknik_ham = son_60_mum[teknik_ozellikler].values
        
        # Makro kol: Sadece en son mumdaki anlık makro durum (Vektör)
        X_makro_ham = son_60_mum[makro_ozellikler].iloc[-1:].values 
        
        # 2. İKİLİ SCALER YÜKLEME VE YÖNLENDİRİCİ PREF BELİRLEME
        kimlik = VARLIK_DNA_DEPOSU.get(sembol, {})
        sektor_dna = kimlik.get("Sektor", "BILINMEYEN")
        pref = sektor_dna if sektor_dna in AI_MODELS else "sp500_genel"
            
        scaler_makro, scaler_teknik = None, None
        
        if pref in AI_MODELS and os.path.exists(f"{pref}_scaler_makro.pkl") and os.path.exists(f"{pref}_scaler_teknik.pkl"):
            import pickle
            with open(f"{pref}_scaler_makro.pkl", "rb") as f:
                scaler_makro = pickle.load(f)
            with open(f"{pref}_scaler_teknik.pkl", "rb") as f:
                scaler_teknik = pickle.load(f)
        else:
            print(f"[UYARI] {pref} için çiftli Scaler dosyaları (veya yüklü model) bulunamadı! Simülasyon modunda devam ediliyor.")
            
        # 3. VERİLERİ ÖLÇEKLENDİR
        if scaler_makro is not None and scaler_teknik is not None:
            try:
                X_makro_olcekli = scaler_makro.transform(X_makro_ham)
                X_teknik_olcekli = scaler_teknik.transform(X_teknik_ham)
            except Exception as e:
                print(f"[UYARI] {sembol} için ölçeklendirme hatası (Eski model/scaler uyumsuzluğu olabilir): {e}")
                X_makro_olcekli = X_makro_ham
                X_teknik_olcekli = X_teknik_ham
        else:
            X_makro_olcekli = X_makro_ham
            X_teknik_olcekli = X_teknik_ham
        
        # 4. TENSÖRE ÇEVİR VE AHTAPOT BEYNE GÖNDER
        olasilik_long = 0.0
        olasilik_short = 0.0
        
        if TORCH_AVAILABLE:
            if scaler_makro is not None and scaler_teknik is not None:
                # X_makro boyutu: (Batch=1, Features=2)
                X_makro_tensor = torch.tensor(X_makro_olcekli, dtype=torch.float32).to(Cihaz)
                
                # X_teknik boyutu: (Batch=1, Seq=60, Features=15)
                X_teknik_tensor = torch.tensor(X_teknik_olcekli, dtype=torch.float32).unsqueeze(0).to(Cihaz)
                
                with torch.no_grad():
                    if pref in AI_MODELS:
                        olasilik_long  = float(torch.sigmoid(AI_MODELS[pref]["LONG"](X_makro_tensor, X_teknik_tensor)).item())
                        olasilik_short = float(torch.sigmoid(AI_MODELS[pref]["SHORT"](X_makro_tensor, X_teknik_tensor)).item())
                    else:
                        olasilik_long = 0.5
                        olasilik_short = 0.5
            else:
                olasilik_long = 0.5
                olasilik_short = 0.5
        else:
            olasilik_long = 0.55
            olasilik_short = 0.45
            
        # Repainting Önleme: son_fiyat canlı en son fiyattır (unclosed), guncel_atr ise son kapalı mumun ATR'sidir.
        son_fiyat = float(df['Close'].iloc[-1])
        guncel_atr = float(son_60_mum['ATR'].iloc[-1])
        
        # LONG SEVİYELERİ
        stop_long_val = son_fiyat - (guncel_atr * 1.5)
        hedef_long_val = son_fiyat + (guncel_atr * 3.0)
        risk_long = son_fiyat - stop_long_val
        odul_long = hedef_long_val - son_fiyat
        rr_long = odul_long / (risk_long + 0.0001)
        
        # SHORT SEVİYELERİ
        stop_short_val = son_fiyat + (guncel_atr * 1.5)
        hedef_short_val = son_fiyat - (guncel_atr * 3.0)
        risk_short = stop_short_val - son_fiyat
        odul_short = son_fiyat - hedef_short_val
        rr_short = odul_short / (risk_short + 0.0001)
        
        # Grafik için gerekli
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        
        return {
            "Sembol": sembol,
            "Ad": ad,
            "Fiyat": round(son_fiyat, 2),
            "Yapay_Zeka_Olasiligi_Long": round(olasilik_long * 100, 2),
            "Yapay_Zeka_Olasiligi_Short": round(olasilik_short * 100, 2),
            "Yapay_Zeka_Olasiligi": round(olasilik_long * 100, 2), # Geriye donuk uyumluluk
            "Skor": int(olasilik_long * 100), # Geriye donuk uyumluluk
            "RR_Orani": round(rr_long, 2),
            "RR_Orani_Long": round(rr_long, 2),
            "RR_Orani_Short": round(rr_short, 2),
            "Stop": round(stop_long_val, 2),
            "Hedef": round(hedef_long_val, 2),
            "Stop_Long": round(stop_long_val, 2),
            "Hedef_Long": round(hedef_long_val, 2),
            "Stop_Short": round(stop_short_val, 2),
            "Hedef_Short": round(hedef_short_val, 2),
            "Onaylar": [f"Yapay Zeka Long Olasilik: %{olasilik_long * 100:.2f}", f"Yapay Zeka Short Olasilik: %{olasilik_short * 100:.2f}"],
            "Veri": df
        }
    except Exception as e:
        print(f"[Hata] {sembol} AI cikarimi yapilamadi: {e}")
        return None

def hiyerarsik_analiz_yap(sembol, ad=""):
    """
    MTFA (Multi-Timeframe Analysis) Motoru:
    Günlük grafikten onay alır, 1 Saatlikten momentumu ölçer, 15 Dakikalıkta tetiği çeker.
    """
    try:
        df_gunluk = veri_deposu.veri_getir(sembol, period="1y", interval="1d")
        df_1saat = veri_deposu.veri_getir(sembol, period="1mo", interval="1h")
        df_15dk = veri_deposu.veri_getir(sembol, period="5d", interval="15m")

        if len(df_gunluk) < 2 or len(df_1saat) < 2 or len(df_15dk) < 2: 
            return None

        for df in [df_gunluk, df_1saat, df_15dk]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

        # --- VETO KONTROLÜ (Skor Sıfırlama Mantığı - Repainting Önleme: son kapalı mumlar kullanılır) ---
        veto_yedi = False
        df_gunluk['EMA_200'] = df_gunluk['Close'].ewm(span=200, adjust=False).mean()
        # Repainting Önleme: Son kapalı günlük mum
        gunluk_son = df_gunluk.iloc[-2]
        
        if gunluk_son['Close'] < gunluk_son['EMA_200']:
            veto_yedi = True # Günlük trend bozuk

        delta = df_1saat['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_1saat['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))
        # Repainting Önleme: Son kapalı saatlik mum
        saatlik_son = df_1saat.iloc[-2]
        
        if saatlik_son['RSI'] < 50:
            veto_yedi = True # Saatlik momentum zayıf

        # --- KATMAN 3: KESKİN NİŞANCI (15 DAKİKALIK TETİK - Repainting Önleme: iloc[-2] kullanılır) ---
        df_15 = df_15dk.copy()
        df_15['EMA_5'] = df_15['Close'].ewm(span=5, adjust=False).mean()
        df_15['EMA_20'] = df_15['Close'].ewm(span=20, adjust=False).mean()
        
        df_15['H-L'] = df_15['High'] - df_15['Low']
        df_15['H-PC'] = abs(df_15['High'] - df_15['Close'].shift(1))
        df_15['L-PC'] = abs(df_15['Low'] - df_15['Close'].shift(1))
        df_15['TR'] = df_15[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df_15['ATR'] = df_15['TR'].rolling(window=14).mean()

        df_15['Stop_Loss'] = df_15['Close'] - (df_15['ATR'] * 1.5)
        df_15['Hedef'] = df_15['Close'] + (df_15['ATR'] * 3.0)     
        
        df_15['Risk_Miktari'] = df_15['Close'] - df_15['Stop_Loss']
        df_15['Odul_Miktari'] = df_15['Hedef'] - df_15['Close']
        df_15['RR_Orani'] = df_15['Odul_Miktari'] / (df_15['Risk_Miktari'] + 0.0001)

        df_15['Gövde'] = abs(df_15['Close'] - df_15['Open'])
        df_15['Alt_Fitil'] = df_15[['Close', 'Open']].min(axis=1) - df_15['Low']
        df_15['Ust_Fitil'] = df_15['High'] - df_15[['Close', 'Open']].max(axis=1)
        df_15['Cekic'] = (df_15['Alt_Fitil'] > (2 * df_15['Gövde'])) & (df_15['Ust_Fitil'] < (0.5 * df_15['Gövde'])) & (df_15['Gövde'] > 0)
        
        c5_tetik = (df_15['EMA_5'] > df_15['EMA_20']) & (df_15['EMA_5'].shift(1) <= df_15['EMA_20'].shift(1))

        # Skorlama (Repainting Önleme: iloc[-2] kapalı mum verisi kullanılır)
        skor = 0
        son_15 = df_15.iloc[-2]
        live_fiyat = float(df_15['Close'].iloc[-1]) # Canlı en son fiyat
        
        if c5_tetik.iloc[-2]: skor += 35
        if son_15['Cekic']: skor += 30
        if son_15['RR_Orani'] >= 2.0: skor += 20
        
        # Eğer büyük zaman dilimlerinden veto yediysek, ne olursa olsun skoru sıfırla
        if veto_yedi:
            skor = 0
            
        return {
            "Sembol": sembol,
            "Ad": ad,
            "Fiyat": round(live_fiyat, 2),
            "Skor": int(skor),
            "RR_Orani": round(float(son_15['RR_Orani']), 2),
            "Stop": round(float(son_15['Stop_Loss']), 2),
            "Hedef": round(float(son_15['Hedef']), 2),
            "Veri": df_15 
        }
    except Exception as e:
        print(f"[Hata] {sembol} MTFA analizi yapilamadi: {e}")
        return None

# ==========================================
# ANA ANALİZ MOTORU (DRY PRENSİBİ)
# ==========================================
def sektorel_para_akisi_hesapla(sektor_etf, ana_endeks="SPY"):
    """
    Sektör ETF'sinin, genel piyasaya (SPY) göre göreceli gücünü (Alpha) hesaplar.
    RS Çizgisi = Sektör Kapanışı / Piyasa Kapanışı.
    Bu çizginin son 5 günlük ivmesi bize paranın yönünü söyler.
    """
    if not sektor_etf or sektor_etf == "BILINMEYEN":
        return 0.0 # Bilinmeyen sektörler için nötr
        
    try:
        # 1. Cache sisteminden son 1 aylık verileri çek (Çok hızlıdır)
        df_sektor = veri_deposu.veri_getir(sektor_etf, period="1mo", interval="1d")
        df_piyasa = veri_deposu.veri_getir(ana_endeks, period="1mo", interval="1d")
        
        if df_sektor.empty or df_piyasa.empty:
            return 0.0
            
        # 2. Tarihleri hizalayarak iki veriyi birleştir
        ortak_df = pd.DataFrame({
            'Sektor_Close': df_sektor['Close'],
            'Piyasa_Close': df_piyasa['Close']
        }).dropna()
        
        if len(ortak_df) < 10:
            return 0.0
            
        # 3. Göreceli Güç (RS) Çizgisini Oluştur
        ortak_df['RS_Line'] = ortak_df['Sektor_Close'] / ortak_df['Piyasa_Close']
        
        # 4. Kısa Vadeli Rotasyon İvmesi (Son 5 Gün)
        # Eğer RS çizgisi son 5 günde yükseliyorsa, para bu sektöre giriyordur (Pozitif Ayrışma)
        rs_momentum = ortak_df['RS_Line'].pct_change(periods=5).iloc[-1]
        
        # Yüzdelik getiri farkı olarak döndür (+%2.5 veya -%1.2)
        return float(rs_momentum * 100)
        
    except Exception as e:
        print(f"[UYARI] {sektor_etf} sektörel para akışı hesaplanamadı: {e}")
        return 0.0

def analiz_yap(sembol, ad, df_vix, df_dxy, df_sektorler=None):
    """Tek bir varlığı önbellekten veya yfinance'ten alır, tüm göstergeleri ve skorları hesaplar."""
    try:
        kimlik = VARLIK_DNA_DEPOSU.get(sembol, None)
        if kimlik is None:
            kimlik = {
                "Sektor": "BILINMEYEN", 
                "Sektor_ETF": "BILINMEYEN",
                "Endustri": "BILINMEYEN",
                "Endustri_ETF": "BILINMEYEN",
                "Karakter": "STANDART", 
                "Makro_Agirlik": {"VIX": 0, "DXY": 0}
            }

        df_hisse = veri_deposu.veri_getir(sembol, period="2y", interval="1d")
        if df_hisse.empty: 
            return None
        
        if isinstance(df_hisse.columns, pd.MultiIndex):
            df_hisse.columns = df_hisse.columns.droplevel(1)
        df_hisse = df_hisse[['Open', 'High', 'Low', 'Close']]
            
        df = df_hisse.join([df_vix, df_dxy]).dropna()
        if len(df) < 2: 
            return None
            
        if df_sektorler:
            for sektor_df in df_sektorler.values():
                df = df.join(sektor_df, how='left')
        
        makro_kolonlar = ['VIX', 'DXY'] + list(SEKTOR_ETFLERI.values())
        for col in makro_kolonlar:
            if col not in df.columns:
                df[col] = 0.0
        df[makro_kolonlar] = df[makro_kolonlar].ffill().bfill().fillna(0.0)

        # Göstergeler & ATR Stop
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['DXY_SMA_20'] = df['DXY'].rolling(window=20).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))

        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # 1. Varlığın DNA'sından ATR Çarpanlarını Çek
        karakter = kimlik.get("Karakter", "OFANSIF_BUYUME")
        
        if karakter == "GUVENLI_LIMAN_VE_HAMMADDE": # Örn: SI=F
            atr_stop_carpani = 2.5
            atr_hedef_carpani = 5.0
        elif karakter == "STABIL_DEFANSIF": # Örn: JNJ, KO
            atr_stop_carpani = 1.2
            atr_hedef_carpani = 2.4
        else: # Standart Hisseler (Örn: AAPL)
            atr_stop_carpani = 1.5
            atr_hedef_carpani = 3.0

        # 2. Her Varlığa Özel Dinamik Seviyelerin Hesaplanması
        df['Stop_Loss'] = df['Close'] - (df['ATR'] * atr_stop_carpani)
        df['Direnc_Hedefi'] = df['Close'] + (df['ATR'] * atr_hedef_carpani)

        df['Risk_Miktari'] = df['Close'] - df['Stop_Loss']
        df['Odul_Miktari'] = df['Direnc_Hedefi'] - df['Close']
        df['RR_Orani'] = df['Odul_Miktari'] / (df['Risk_Miktari'] + 0.0001)

        df['Gövde'] = abs(df['Close'] - df['Open'])
        df['Alt_Fitil'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['Ust_Fitil'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['Cekic_Formasyonu'] = (df['Alt_Fitil'] > (2 * df['Gövde'])) & (df['Ust_Fitil'] < (0.5 * df['Gövde'])) & (df['Gövde'] > 0)
        
        dun_kirmizi = df['Close'].shift(1) < df['Open'].shift(1)
        bugun_yesil = df['Close'] > df['Open']
        govdeyi_yuttu = (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))
        df['Yutan_Boga'] = dun_kirmizi & bugun_yesil & govdeyi_yuttu

        df['Eski_Dip'] = df['Low'].shift(15).rolling(window=25).min()
        df['Yeni_Dip'] = df['Low'].rolling(window=10).min()
        esit_dipler = (abs(df['Eski_Dip'] - df['Yeni_Dip']) / df['Eski_Dip']) <= 0.03
        df['Boyun_Cizgisi'] = df['High'].shift(5).rolling(window=15).max()
        df['Ikili_Dip_Formasyonu'] = esit_dipler & (df['Close'] > df['Boyun_Cizgisi']) & (df['Close'].shift(1) <= df['Boyun_Cizgisi'].shift(1))

        # Skorlama
        df['Toplam_Puan'] = 0
        df.loc[df['VIX'] < 25.0, 'Toplam_Puan'] += 10
        df.loc[df['DXY'] < df['DXY_SMA_20'], 'Toplam_Puan'] += 10
        df.loc[df['Close'] > df['EMA_200'], 'Toplam_Puan'] += 20
        df.loc[(df['RSI'] > 35) & (df['RSI'] < 75), 'Toplam_Puan'] += 15
        
        c5_tetik = (df['EMA_5'] > df['EMA_20']) & (df['EMA_5'].shift(1) <= df['EMA_20'].shift(1))
        df.loc[c5_tetik, 'Toplam_Puan'] += 15
        
        df.loc[df['Cekic_Formasyonu'], 'Toplam_Puan'] += 15
        df.loc[df['Yutan_Boga'], 'Toplam_Puan'] += 20
        df.loc[df['Ikili_Dip_Formasyonu'] == True, 'Toplam_Puan'] += 30

        df.loc[df['RR_Orani'] >= 2.0, 'Toplam_Puan'] += 30
        df.loc[df['RR_Orani'] < 1.0, 'Toplam_Puan'] = 0  # VETO



        # ==========================================
        # 2. YAPAY ZEKA GÜVEN EŞİKLERİNİ DİNAMİK AYARLA
        # ==========================================
        if kimlik["Sektor"] == "EMTIA_VE_MADENCILIK":
            AI_AL_ESIGI = 65.0
        elif kimlik["Karakter"] == "OFANSIF_BUYUME":
            AI_AL_ESIGI = 58.0
        else:
            AI_AL_ESIGI = 60.0

        # ==========================================
        # 3. DİNAMİK MAKRO PUANLAMA (DNA'ya Göre Puan Ekleme/Çıkarma)
        # ==========================================
        vix_hassasiyeti = kimlik["Makro_Agirlik"].get("VIX", 0)
        dxy_hassasiyeti = kimlik["Makro_Agirlik"].get("DXY", 0)
        
        # Eğer piyasada korku (VIX) azsa ve bu hisse hücum hissesiyse (Negatif VIX hassasiyeti)
        if vix_hassasiyeti < 0:
            df.loc[df['VIX'] < 20.0, 'Toplam_Puan'] += 15 # Teknolojiye Bonus!
        elif vix_hassasiyeti > 0:
            df.loc[df['VIX'] < 20.0, 'Toplam_Puan'] -= 10 # Altından Puan Kes!
            
        # Eğer piyasada korku (VIX) yüksekse
        if vix_hassasiyeti > 0:
            df.loc[df['VIX'] > 25.0, 'Toplam_Puan'] += 20 # Altına Çılgın Bonus!
        elif vix_hassasiyeti < 0:
            df.loc[df['VIX'] > 25.0, 'Toplam_Puan'] -= 20 # Teknolojiyi Veto Et!

        # ==========================================
        # 4. SEKTÖREL PARA AKIŞI (ROTASYON) FİLTRESİ
        # ==========================================
        sektor_etf = kimlik.get("Sektor_ETF", "")
        para_akisi_skoru = sektorel_para_akisi_hesapla(sektor_etf)
        
        if para_akisi_skoru > 1.0: # %1'den büyük pozitif ayrışma
            df['Toplam_Puan'] += 25
        elif para_akisi_skoru > 0.0:
            df['Toplam_Puan'] += 10
        elif para_akisi_skoru < -1.0: # %1'den büyük negatif ayrışma
            df['Toplam_Puan'] -= 25 # Yapay zeka sinyali bile olsa VETO'ya yaklaştırır

        # Repainting Önleme: Analiz ve formasyonlar son kapalı günlük mum (iloc[-2]) ile yapılır
        son_gun = df.iloc[-2]
        live_fiyat = float(df['Close'].iloc[-1]) # Canlı en son fiyat
        onaylar = []
        if son_gun['Cekic_Formasyonu']: onaylar.append("Çekiç Formasyonu")
        if son_gun['Yutan_Boga']: onaylar.append("Yutan Boğa")
        if son_gun['Ikili_Dip_Formasyonu']: onaylar.append("İkili Dip (W)")

        # Sektörel para akışı onaylarını ekleyelim
        if para_akisi_skoru > 1.0:
            onaylar.append(f"Sektörel Para Girişi (+%{para_akisi_skoru:.1f} Alpha)")
        elif para_akisi_skoru < -1.0:
            onaylar.append(f"⚠️ Sektörel Para Çıkışı (-%{abs(para_akisi_skoru):.1f} Zayıflık)")

        return {
            "Sembol": sembol,
            "Ad": ad,
            "Fiyat": round(live_fiyat, 2),
            "Skor": min(100, max(0, int(son_gun['Toplam_Puan']))),
            "AI_AL_ESIGI": AI_AL_ESIGI,
            "RR_Orani": round(float(son_gun['RR_Orani']), 2),
            "Stop": round(float(son_gun['Stop_Loss']), 2),
            "Hedef": round(float(son_gun['Direnc_Hedefi']), 2),
            "Onaylar": onaylar,
            "Veri": df
        }
    except Exception as e:
        print(f"[Hata] {sembol} analizi yapilirken hata: {e}")
        return None

# ==========================================
# ABD EN BÜYÜK 100 ŞİRKET LİSTESİ
# ==========================================
us100_hisseleri = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "TSLA",
    "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV",
    "MRK", "NFLX", "AMD", "CRM", "CVX", "WMT", "ADBE", "KO", "BAC", "PEP",
    "TMO", "ACN", "DIS", "CSCO", "QCOM", "LIN", "ORCL", "INTC", "MCD", "TXN",
    "ABT", "INTU", "GE", "CAT", "AMGN", "VZ", "PFE", "PM", "MS", "IBM",
    "CMCSA", "AXP", "GS", "UNP", "SPGI", "COP", "HON", "AMAT", "ISRG", "BKNG",
    "LOW", "SBUX", "MDLZ", "PLTR", "VRTX", "ADI", "LRCX", "TJX", "REGN", "SYK",
    "ELV", "GILD", "PANW", "MU", "CDNS", "SNPS", "KLAC", "ADP", "CI", "C",
    "MAR", "CRWD", "BSX", "HCA", "DE", "LMT", "MDT", "FTNT", "ORLY", "CTAS",
    "MELI", "CSX", "ITW", "WM", "GD", "WDAY", "EOG", "MCK", "HUM", "ABNB"
]

# ==========================================
# UYGULAMA AKIŞI (MOD YÖNETİMİ)
# ==========================================
def calistir_gece_taramasi():
    """
    FAZ 1: GECE TARAMASI (Tüm Liste Taranır -> Aday Havuzu JSON)
    """
    print("\n=== FAZ 1 BASLADI: Tum Piyasa Taraniyor (Gunde 1 Kez) ===")
    aday_havuzu = {}
    
    # Tüm varlık listesini VARLIK_DNA_DEPOSU'ndan dinamik olarak oluştur
    tarama_listesi = []
    metaller = {}
    
    if TEST_MODU:
        tarama_listesi = ["AAPL", "MSFT"]
        metaller = {
            "GC=F": "ALTIN",
            "SI=F": "GUMUS"
        }
    else:
        # Varlık listesini DNA deposundan otomatik ayrıştır
        for sembol, dna in VARLIK_DNA_DEPOSU.items():
            if dna["Sektor"] == "EMTIA_VE_MADENCILIK":
                # Ad olarak endüstri adını (örneğin KIYMETLI_MADENLER, ENERJI_EMTIALARI) veya sembolü kullan
                metaller[sembol] = dna.get("Endustri", sembol)
            else:
                tarama_listesi.append(sembol)
                
        # Eğer veri tabanı boşsa fallback olarak us100_hisseleri ve default emtiaları kullan
        if not tarama_listesi:
            tarama_listesi = us100_hisseleri.copy()
            metaller = {
                "GC=F": "ALTIN",
                "SI=F": "GUMUS"
            }
    
    # 1. Toplu yfinance indirme listesi oluştur
    bulk_symbols = tarama_listesi + list(metaller.keys()) + ["^VIX", "DX-Y.NYB", "SPY"] + list(SEKTOR_ETFLERI.keys())
    print(f"[BULK] {len(bulk_symbols)} enstrüman için veriler toplu olarak indiriliyor...")
    
    try:
        df_bulk = yf.download(bulk_symbols, period="2y", interval="1d", group_by="ticker", progress=False)
        if df_bulk is not None and not df_bulk.empty:
            for sembol in bulk_symbols:
                try:
                    # yfinance multi-ticker download creates a MultiIndex where level 0 is the ticker symbol
                    if isinstance(df_bulk.columns, pd.MultiIndex) and sembol in df_bulk.columns.levels[0]:
                        df_sym = df_bulk[sembol].dropna(how='all')
                        if not df_sym.empty:
                            veri_deposu.veri_getir(sembol, period="2y", interval="1d", bulk_df=df_sym)
                    elif sembol in df_bulk.columns:
                        # Single ticker fallback if it downloaded as a standard DataFrame
                        df_sym = df_bulk[[sembol]].dropna(how='all')
                        if not df_sym.empty:
                            veri_deposu.veri_getir(sembol, period="2y", interval="1d", bulk_df=df_sym)
                except Exception as e:
                    print(f"[UYARI] {sembol} önbelleğe yazılamadı: {e}")
    except Exception as e:
        print(f"[HATA] Toplu indirme başarısız oldu: {e}. Tekil indirme moduna geçilecek...")

    # 2. Güncel VIX ve DXY'yi cache'ten oku
    df_vix_raw = veri_deposu.veri_getir("^VIX", period="2y", interval="1d")
    df_vix = df_vix_raw[['Close']].rename(columns={'Close': 'VIX'})
    df_dxy_raw = veri_deposu.veri_getir("DX-Y.NYB", period="2y", interval="1d")
    df_dxy = df_dxy_raw[['Close']].rename(columns={'Close': 'DXY'})
    
    # 3. REJİM KALKANI (Ayı Piyasası Filtresi)
    ayi_piyasasi = False
    spy_fiyati = 0
    spy_ema200 = 0
    vix_fiyati = df_vix_raw['Close'].iloc[-1] if not df_vix_raw.empty else 0
    df_spy_raw = veri_deposu.veri_getir("SPY", period="2y", interval="1d")
    
    if not df_spy_raw.empty:
        df_spy_raw['EMA_200'] = df_spy_raw['Close'].ewm(span=200, adjust=False).mean()
        spy_fiyati = float(df_spy_raw['Close'].iloc[-1])
        spy_ema200 = float(df_spy_raw['EMA_200'].iloc[-1])
        
        if spy_fiyati < spy_ema200 and vix_fiyati > 25:
            ayi_piyasasi = True
            print(f"🚨 [DİKKAT] AYI PİYASASI REJİMİ AKTİF! (SPY: {spy_fiyati:.2f} < EMA200: {spy_ema200:.2f} | VIX: {vix_fiyati:.2f} > 25)")
            
    dosya_yaz("piyasa_durumu.json", {
        "Ayi_Piyasasi": ayi_piyasasi,
        "SPY_Fiyat": spy_fiyati,
        "SPY_EMA200": spy_ema200,
        "VIX_Fiyat": float(vix_fiyati)
    })
    
    df_sektorler = {}
    for etf, s_ad in SEKTOR_ETFLERI.items():
        df_s_raw = veri_deposu.veri_getir(etf, period="2y", interval="1d")
        if not df_s_raw.empty:
            df_s = df_s_raw[['Close']].pct_change(periods=5) * 100.0
            df_sektorler[s_ad] = df_s.rename(columns={'Close': s_ad})

    # Metalleri Tara (SQLite cache sayesinde milisaniyede tamamlanır!)
    for sembol, ad in metaller.items():
        try:
            print(f"[METAL] Analiz ediliyor: {ad} ({sembol})...")
            sonuc = analiz_yap(sembol, ad, df_vix, df_dxy, df_sektorler)
            if sonuc and sonuc["Skor"] >= 70:
                aday_havuzu[sembol] = sonuc
        except Exception as e:
            print(f"[Hata] {ad} gece taramasinda hata: {e}")

    # Hisseleri Tara (SQLite cache sayesinde milisaniyede tamamlanır!)
    toplam_hisse = len(tarama_listesi)
    for i, sembol in enumerate(tarama_listesi, 1):
        try:
            print(f"[{i}/{toplam_hisse}] Analiz ediliyor: {sembol}...", flush=True)
            sonuc = analiz_yap(sembol, sembol, df_vix, df_dxy, df_sektorler)
            if sonuc and sonuc["Skor"] >= 70:
                aday_havuzu[sembol] = sonuc
        except Exception as e:
            print(f"\n[Hata] {sembol} gece taramasinda hata: {e}")

    # Aday havuzu JSON dosyasına kaydet
    dosya_yaz(ADAY_DOSYASI, aday_havuzu)
    
    # Telegram Bildirimi Hazırla ve Gönder
    bugunun_tarihi = time.strftime('%d-%m-%Y')
    test_etiketi = " [TEST RAPORU]" if TEST_MODU else ""
    mesaj = (
        f"🌙 *MASTER BOT: GECE TARAMASI TAMAMLANDI*{test_etiketi} 🌙\n\n"
        f"📅 *Tarih:* {bugunun_tarihi}\n"
        f"🔍 *Taranan Enstrüman:* {toplam_hisse + len(metaller)}\n"
        f"📈 *Takibe Alınan Aday (70+ Puan):* {len(aday_havuzu)} adet\n\n"
    )
    if aday_havuzu:
        mesaj += "📋 *Aday Listesi & Skorlar (İlk 25 Varlık):*\n"
        mesaj += "="*30 + "\n"
        sirali_adaylar = sorted(aday_havuzu.values(), key=lambda x: x['Skor'], reverse=True)
        for a in sirali_adaylar[:25]:
            mesaj += f"• *{a['Sembol']} ({a['Ad']})* | Skor: **%{a['Skor']}** | Fiyat: ${a['Fiyat']:.2f}\n"
            
        if len(sirali_adaylar) > 25:
            mesaj += f"\n... ve diğer *{len(sirali_adaylar) - 25}* aday.\n*(Tüm aday listesi '{ADAY_DOSYASI}' dosyasına kaydedilmiştir.)*\n"
    else:
        mesaj += "Bugün 70+ barajını aşan hiçbir aday bulunamadı."
        
    # Kullanıcı isteği: Aday havuzu Telegram mesajını gönderme.
    # send_telegram_message(mesaj) 
    print(f"\nGece taramasi basariyla tamamlandi. Aday havuzu '{ADAY_DOSYASI}' dosyasina kaydedildi.")

    # OPSIYON VERI AMBARINI GUNCELLE (Sadece Gece Taramasindan Sonra Calisir)
    print("\n[VERI AMBARI] Opsiyon Arşivi Güncelleniyor...")
    try:
        from opsiyon_ambari import ambar_guncelle
        ambar_guncelle(bulk_symbols)
    except Exception as e:
        print(f"[UYARI] Opsiyon Ambarı güncellenirken hata oluştu: {e}")


def calistir_gun_ici_kontrol():
    """
    FAZ 2 & 3: GÜN İÇİ KONTROL (Sadece Adaylar ve Aktif Portföy)
    """
    print("\n=== FAZ 2 & 3 BASLADI: Gun Ici Portfoy ve Aday Havuzu Kontrolu ===")
    aday_havuzu = dosya_oku(ADAY_DOSYASI)
    aktif_portfoy = dosya_oku(PORTFOY_DOSYASI)
    
    # Test modunda mock verilerle simülasyon yapalım
    if TEST_MODU and not aday_havuzu:
        print("[TEST] Aday havuzu bos oldugundan test icin mock veriler ekleniyor...")
        aday_havuzu = {
            "AAPL": {
                "Sembol": "AAPL",
                "Ad": "AAPL",
                "Fiyat": 180.0,
                "Skor": 75,  # Alım tetikleyecek (>= 70)
                "RR_Orani": 2.2,
                "Stop": 172.0,
                "Hedef": 198.0,
                "Onaylar": ["İkili Dip (W) Formasyonu"]
            },
            "MSFT": {
                "Sembol": "MSFT",
                "Ad": "MSFT",
                "Fiyat": 420.0,
                "Skor": 65,  # Havuzdan atılacak (< 70)
                "RR_Orani": 2.5,
                "Stop": 405.0,
                "Hedef": 457.5,
                "Onaylar": []
            }
        }
        dosya_yaz(ADAY_DOSYASI, aday_havuzu)
        
    test_etiketi = " [TEST RAPORU]" if TEST_MODU else ""
    rapor_mesaji = f"☀️ *MASTER BOT: GÜN İÇİ PİYASA RAPORU*{test_etiketi} ☀️\n\n"
    
    # --- FAZ 3: AKTİF PORTFÖYÜN SAĞLIK KONTROLÜ ---
    silinecek_portfoy = []
    is_portfoy_changed = False
    
    # Kalan işlemlerin Sağlık Kontrolü (1m ve MTFA analizi ile)
    if aktif_portfoy:
        rapor_mesaji += "💼 *AKTİF PORTFÖY DURUMU:*\n"
        rapor_mesaji += "="*30 + "\n"
        for sembol, portfoy_verisi in list(aktif_portfoy.items()):
            try:
                time.sleep(random.uniform(1.0, 2.0))
                print(f"[PORTFOY] Kontrol ediliyor: {sembol}...")
                
                stop_seviyesi = portfoy_verisi["Stop"]
                hedef_seviyesi = portfoy_verisi["Hedef"]
                yon = portfoy_verisi.get("Yon", "LONG")
                
                # 1. BOT UYURKEN NE OLDU? (1 Dakikalık Röntgen)
                gecmis_durum = on_bes_dakikalik_gecmis_kontrol(sembol, stop_seviyesi, hedef_seviyesi, yon)
                
                giris_fiyati = portfoy_verisi.get("Fiyat", stop_seviyesi)
                adet = portfoy_verisi.get("Adet", 1)

                if gecmis_durum == "STOP_VURULDU":
                    rapor_mesaji += f"🛑 *{sembol} {yon} POZİSYONU STOP OLDU!* \nZarar kesildi. (Çıkış: ${stop_seviyesi:.2f})\n"
                    satis_islemi_gerceklestir(sembol, giris_fiyati, stop_seviyesi, adet, "STOP-LOSS", yon)
                    aktif_portfoy.pop(sembol, None)  # Çift satışı önlemek için önce sil
                    is_portfoy_changed = True
                    dosya_yaz(PORTFOY_DOSYASI, aktif_portfoy) # Anlık UI güncellemesi
                    continue
                    
                elif gecmis_durum == "HEDEF_VURULDU":
                    rapor_mesaji += f"✅ *{sembol} {yon} POZİSYONU HEDEF VURDU!* \nKar cebe atıldı. (Çıkış: ${hedef_seviyesi:.2f})\n"
                    satis_islemi_gerceklestir(sembol, giris_fiyati, hedef_seviyesi, adet, "TAKE-PROFIT", yon)
                    aktif_portfoy.pop(sembol, None)  # Çift satışı önlemek için önce sil
                    is_portfoy_changed = True
                    dosya_yaz(PORTFOY_DOSYASI, aktif_portfoy) # Anlık UI güncellemesi
                    continue

                # 2. HİÇBİRİNE DEĞMEDİYSE MEVCUT DURUMA BAK
                hisse_adi = portfoy_verisi.get("Ad", sembol)
                guncel = yapay_zeka_ile_analiz_yap(sembol, hisse_adi)
                if guncel:
                    ai_olasiligi_long = guncel["Yapay_Zeka_Olasiligi_Long"]
                    ai_olasiligi_short = guncel["Yapay_Zeka_Olasiligi_Short"]
                    guncel_atr = guncel["Veri"]['ATR'].iloc[-1]
                    guncel_fiyat = guncel["Fiyat"]
                    
                    # Giriş tarihini ve geçen gün sayısını hesapla (Zaman aşımı için)
                    if "Giris_Tarihi" not in portfoy_verisi:
                        portfoy_verisi["Giris_Tarihi"] = datetime.now().strftime("%Y-%m-%d")
                        is_portfoy_changed = True
                    
                    if yon == "LONG":
                        SAT_ESIGI = 10.0
                    else:
                        SAT_ESIGI = 10.0
                    
                    giris_tarihi_str = portfoy_verisi["Giris_Tarihi"]
                    try:
                        giris_tarihi_dt = datetime.strptime(giris_tarihi_str, "%Y-%m-%d")
                        gecen_gun = (datetime.now() - giris_tarihi_dt).days
                    except Exception:
                        gecen_gun = 0

                    # Eşikleri belirle (Dinamik)
                    kimlik = VARLIK_DNA_DEPOSU.get(sembol, None)
                    if kimlik is None:
                        kimlik = {
                            "Sektor": "BILINMEYEN",
                            "Karakter": "STANDART"
                        }
                    
                    if kimlik["Sektor"] == "EMTIA_VE_MADENCILIK":
                        AL_ESIGI = 78.0
                        SAT_ESIGI = 55.0
                    elif kimlik["Karakter"] == "OFANSIF_BUYUME":
                        AL_ESIGI = 75.0
                        SAT_ESIGI = 55.0
                    else:
                        AL_ESIGI = 76.0
                        SAT_ESIGI = 55.0
                        
                    if not AI_MODELS:
                        # Fallback for mock simulation mode
                        AL_ESIGI = 80.0
                        SAT_ESIGI = 45.0

                    # --- 1. AI ÖZGÜVEN KAYBI VEYA REVERSAL (ERKEN ÇIKIŞ) ---
                    if yon == "LONG":
                        if ai_olasiligi_long < SAT_ESIGI or ai_olasiligi_short >= AL_ESIGI:
                            sebep = "AI ÖZGÜVEN KAYBI" if ai_olasiligi_long < SAT_ESIGI else f"AI REVERSAL SİNYALİ (SHORT %{ai_olasiligi_short:.1f})"
                            rapor_mesaji += f"⚠️ *{sembol} LONG POZİSYONU ERKEN KAPATILDI!* ({sebep}. Fiyat: ${guncel_fiyat:.2f})\n"
                            satis_islemi_gerceklestir(sembol, giris_fiyati, guncel_fiyat, adet, sebep, yon)
                            silinecek_portfoy.append(sembol)
                            is_portfoy_changed = True
                            dosya_yaz(PORTFOY_DOSYASI, aktif_portfoy) # Anlık UI güncellemesi
                            continue
                    else: # SHORT
                        if ai_olasiligi_short < SAT_ESIGI or ai_olasiligi_long >= AL_ESIGI:
                            sebep = "AI ÖZGÜVEN KAYBI" if ai_olasiligi_short < SAT_ESIGI else f"AI REVERSAL SİNYALİ (LONG %{ai_olasiligi_long:.1f})"
                            rapor_mesaji += f"⚠️ *{sembol} SHORT POZİSYONU ERKEN KAPATILDI!* ({sebep}. Fiyat: ${guncel_fiyat:.2f})\n"
                            satis_islemi_gerceklestir(sembol, giris_fiyati, guncel_fiyat, adet, sebep, yon)
                            silinecek_portfoy.append(sembol)
                            is_portfoy_changed = True
                            continue

                    # ==========================================
                    # 2. KADEMELİ DİNAMİK İZLEYEN STOP (Step-Up Trend Sömürücüsü)
                    # ==========================================
                    eski_stop = portfoy_verisi["Stop"]
                    guncel_stop, aktif_carpan, vites_durumu = dinamik_kademeli_trailing_stop(
                        yon, giris_fiyati, guncel_fiyat, eski_stop, guncel_atr
                    )
                    
                    if guncel_stop != eski_stop:
                        portfoy_verisi["Stop"] = guncel_stop
                        is_portfoy_changed = True
                        rapor_mesaji += f"📈 *{sembol} VİTES: {vites_durumu}* (İzleyen stop yukarı çekildi: ${guncel_stop:.2f})\n"
                        
                    if yon == "LONG" and guncel_fiyat <= guncel_stop:
                        satis_islemi_gerceklestir(sembol, giris_fiyati, guncel_fiyat, adet, "TRAILING_STOP_TETIKLENDI", yon)
                        silinecek_portfoy.append(sembol)
                        is_portfoy_changed = True
                        continue
                    elif yon == "SHORT" and guncel_fiyat >= guncel_stop:
                        satis_islemi_gerceklestir(sembol, giris_fiyati, guncel_fiyat, adet, "TRAILING_STOP_TETIKLENDI", yon)
                        silinecek_portfoy.append(sembol)
                        is_portfoy_changed = True
                        continue

                    ai_olasiligi = ai_olasiligi_long if yon == "LONG" else ai_olasiligi_short
                    rapor_mesaji += f"⏳ {sembol} ({yon}) tutuluyor. (Güncel Fiyat: ${guncel_fiyat:.2f} | AI: %{ai_olasiligi:.2f} | Gün: {gecen_gun})\n"
            except Exception as e:
                print(f"[Hata] Portfoydeki {sembol} gun ici taranirken hata olustu: {e}")
        
        # Satılan veya kapanan işlemleri aktif portföyden temizle
        for s in silinecek_portfoy:
            aktif_portfoy.pop(s, None)
    else:
        rapor_mesaji += "💼 Aktif taşınan bir pozisyon bulunmuyor.\n\n"

    # --- FAZ 2: ADAY HAVUZU KONTROLÜ (Giriş Kararları) ---
    MAX_ISLEM_SAYISI = 5
    
    silinecek_adaylar = []
    is_adaylar_changed = False
    kac_tane_listelendi = 0
    
    hesap = hesap_bilgisi_getir()
    sistem_kilitli = hesap.get("Sistem_Kilitli", False)
    devre_kesici_aktif = hesap.get("Devre_Kesici_Bitis", "2000-01-01") > datetime.now().strftime("%Y-%m-%d")
    
    # Piyasa durumunu oku (Ayi_Piyasasi)
    piyasa_durumu = dosya_oku("piyasa_durumu.json")
    ayi_piyasasi = piyasa_durumu.get("Ayi_Piyasasi", False)
    
    if sistem_kilitli:
        rapor_mesaji += f"🚨 *SİSTEM KİLİTLİ!* (ATH Guard: Zirveden %12 düşüş yaşandı. İşlem açılamaz.)\n\n"
        print("[BİLGİ] Sistem kilitli. Yeni alım yapılmayacak.")
    elif devre_kesici_aktif:
        rapor_mesaji += f"⚠️ *DEVRE KESİCİ AKTİF!* (Maksimum kayıp limiti aşıldı. Bitiş: {hesap['Devre_Kesici_Bitis']})\n\n"
        print(f"[BİLGİ] Devre kesici aktif ({hesap['Devre_Kesici_Bitis']}). Yeni alım yapılmayacak.")
    elif len(aktif_portfoy) >= MAX_ISLEM_SAYISI:
        rapor_mesaji += f"⚠️ *AKTİF İŞLEM SINIRINA (HARD CAP) ULAŞILDI!* ({len(aktif_portfoy)}/{MAX_ISLEM_SAYISI})\n"
        rapor_mesaji += "Sistem kasanın en az %30'unu nakitte (Buffer) tutmak ve fırsat maliyetini korumak için yeni işlem açmayacaktır.\n\n"
        print(f"[BİLGİ] Maksimum islem sinirina ulasildi ({MAX_ISLEM_SAYISI}). Yeni alim yapilmayacak.")
    elif aday_havuzu:
        rapor_mesaji += "🔍 *ADAY HAVUZU TAKİBİ (MTFA Motoru):*\n"
        rapor_mesaji += "="*30 + "\n"
        # Kara listeyi oku
        bugun = datetime.now().strftime("%Y-%m-%d")
        kara_liste = dosya_oku(KARA_LISTE_DOSYASI) if os.path.exists(KARA_LISTE_DOSYASI) else {}

        for sembol, eski_veri in list(aday_havuzu.items()):
            # --- YENİ: REVENGE TRADE (KARA LİSTE) KONTROLÜ ---
            if sembol in kara_liste and kara_liste[sembol] == bugun:
                rapor_mesaji += f"🛡️ *Revenge Trade Kalkanı:* {sembol} bugün zararla kapandığı için tekrar alınmayacak.\n"
                print(f"[KALKAN] {sembol} kara listede olduğu için es geçildi.")
                silinecek_adaylar.append(sembol)
                is_adaylar_changed = True
                continue
            # -------------------------------------------------
            
            # Eğer aday zaten portföydeyse aday havuzundan atalım
            if sembol in aktif_portfoy:
                silinecek_adaylar.append(sembol)
                is_adaylar_changed = True
                continue
            
            try:
                time.sleep(random.uniform(1.0, 2.0))
                print(f"[ADAY] Kontrol ediliyor: {sembol}...")
                
                hisse_adi = eski_veri.get("Ad", sembol)
                guncel = yapay_zeka_ile_analiz_yap(sembol, hisse_adi)
                
                # Test modunda yfinance patlarsa veya mock veri test ediliyorsa mock verileri doldur
                if not guncel and TEST_MODU:
                    guncel = {
                        "Sembol": sembol,
                        "Fiyat": eski_veri["Fiyat"],
                        "Skor": eski_veri["Skor"],
                        "Yapay_Zeka_Olasiligi_Long": eski_veri.get("Yapay_Zeka_Olasiligi_Long", eski_veri.get("Skor", 70.0)),
                        "Yapay_Zeka_Olasiligi_Short": eski_veri.get("Yapay_Zeka_Olasiligi_Short", 10.0),
                        "Yapay_Zeka_Olasiligi": eski_veri.get("Yapay_Zeka_Olasiligi", eski_veri.get("Skor", 70.0)),
                        "RR_Orani": eski_veri["RR_Orani"],
                        "Stop": eski_veri["Stop"],
                        "Hedef": eski_veri["Hedef"],
                        "Veri": pd.DataFrame()
                    }
                
                if not guncel: 
                    continue
                
                if TEST_MODU and sembol == "AAPL":
                    guncel["Yapay_Zeka_Olasiligi_Long"] = 75.0
                    guncel["Yapay_Zeka_Olasiligi_Short"] = 12.0
                    guncel["Yapay_Zeka_Olasiligi"] = 75.0
                    guncel["Skor"] = 75
                    # AAPL için mock veri çerçevesi yoksa yfinance'den çek
                    if guncel["Veri"].empty:
                        df_test = yf.download("AAPL", period="5d", interval="15m", progress=False)
                        if isinstance(df_test.columns, pd.MultiIndex):
                            df_test.columns = df_test.columns.droplevel(1)
                        # EMA ve göstergeleri hesapla ki grafik çiziminde çökmesin
                        df_test['EMA_200'] = df_test['Close'].ewm(span=200, adjust=False).mean()
                        df_test['EMA_5'] = df_test['Close'].ewm(span=5, adjust=False).mean()
                        df_test['EMA_20'] = df_test['Close'].ewm(span=20, adjust=False).mean()
                        delta = df_test['Close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        df_test['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))
                        guncel["Veri"] = df_test
                
                ai_olasiligi_long = guncel.get("Yapay_Zeka_Olasiligi_Long", guncel["Yapay_Zeka_Olasiligi"])
                ai_olasiligi_short = guncel.get("Yapay_Zeka_Olasiligi_Short", 0.0)
                fiyat = guncel["Fiyat"]
                
                # Eşikler (Dinamik)
                kimlik = VARLIK_DNA_DEPOSU.get(sembol, None)
                if kimlik is None:
                    kimlik = {
                        "Sektor": "BILINMEYEN",
                        "Karakter": "STANDART"
                    }
                
                if kimlik["Sektor"] == "EMTIA_VE_MADENCILIK":
                    AL_ESIGI = 78.0
                elif kimlik["Karakter"] == "OFANSIF_BUYUME":
                    AL_ESIGI = 75.0
                else:
                    AL_ESIGI = 76.0
                    
                if not AI_MODELS:
                    AL_ESIGI = 80.0
                
                # MAKRO TREND FİLTRESİ (200 Günlük Ortalamanın altındayken Long, üzerindeyken Short YASAK)
                try:
                    son_veri = guncel["Veri"].iloc[-1]
                    spot_fiyat = float(son_veri["Close"])
                    ema200 = float(son_veri["EMA_200"])
                    ema20 = float(son_veri["EMA_20"])
                    rsi_degeri = float(son_veri["RSI"])
                    adx_degeri = float(son_veri.get("ADX", 25.0))
                    hacim_anomali_degeri = float(son_veri.get("Hacim_Anomalisi", 0.0))
                    hacim_anomalisi_var_mi = True if hacim_anomali_degeri > 0 else False
                except Exception as e:
                    spot_fiyat = fiyat
                    ema200 = fiyat
                    ema20 = fiyat
                    rsi_degeri = 50.0
                    adx_degeri = 25.0
                    hacim_anomalisi_var_mi = False
                
                izinler = yon_izni_ve_reversion_filtresi(spot_fiyat, ema200, ema20, rsi_degeri, sapma_esigi=0.20)
                spy_trend_long_ok = izinler["LONG"]
                spy_trend_short_ok = izinler["SHORT"]
                rejim_mesaji = izinler["Rejim"]
                risk_carpani = izinler.get("Risk_Carpani", 1.0)
                
                # Giriş Kararı ve Telemetri
                islem_yonu = None
                anlik_aksiyon = "BEKLE"
                
                # ==========================================
                # PRIMARY MODEL: MASTER AI KARAR MOTORU
                # ==========================================
                makro_spy_trend = "AYI" if ayi_piyasasi else "BOGA"
                hisse_sektoru = kimlik.get("Sektor", "BILINMEYEN")
                
                ai_karari, sinyal_sebebi = master_ai_karar_motoru(
                    prob_long=ai_olasiligi_long,
                    prob_short=ai_olasiligi_short,
                    spot_fiyat=spot_fiyat,
                    ema200=ema200,
                    adx_degeri=adx_degeri,
                    hisse_sektoru=hisse_sektoru,
                    makro_spy_trend=makro_spy_trend,
                    esik_long=AL_ESIGI,
                    hacim_anomalisi_var_mi=hacim_anomalisi_var_mi
                )
                
                if ai_karari == "LONG" and spy_trend_long_ok:
                    islem_yonu = "LONG"
                    ai_olasiligi = ai_olasiligi_long
                    anlik_aksiyon = "LONG_GIRIS"
                    stop = guncel.get("Stop_Long", guncel["Stop"])
                    hedef = guncel.get("Hedef_Long", guncel["Hedef"])
                elif ai_karari == "SHORT" and spy_trend_short_ok:
                    islem_yonu = "SHORT"
                    ai_olasiligi = ai_olasiligi_short
                    anlik_aksiyon = "SHORT_GIRIS"
                    stop = guncel.get("Stop_Short", fiyat)
                    hedef = guncel.get("Hedef_Short", fiyat)
                else:
                    anlik_aksiyon = "PAS"
                    if (ai_olasiligi_long > 52.0 or ai_olasiligi_short > 52.0) and kac_tane_listelendi < 3:
                        rapor_mesaji += f"⚠️ *{sembol} MASTER VETO!* ({sinyal_sebebi})\n"
                    continue
                # --- KARA KUTU TELEMETRİ KAYDI ---
                try:
                    df_veri = guncel.get("Veri")
                    if df_veri is not None and not df_veri.empty and not TEST_MODU:
                        son_satir = df_veri.iloc[-1]
                        islem_id = canli_durumu_kara_kutuya_yaz(
                            sembol=sembol,
                            ai_long=float(ai_olasiligi_long),
                            ai_short=float(ai_olasiligi_short),
                            aksiyon=anlik_aksiyon,
                            fiyat=float(fiyat),
                            teknik_dict=son_satir.to_dict(),
                            makro_dict=son_satir.to_dict()
                        )
                        if islem_id:
                            guncel["Telemetri_ID"] = islem_id
                except Exception as e:
                    print(f"[UYARI] {sembol} için telemetri kaydı yapılamadı: {e}")

                if islem_yonu is not None:
                    # REJİM KALKANI VETOSU
                    if islem_yonu == "LONG" and ayi_piyasasi and kimlik["Sektor"] != "EMTIA_VE_MADENCILIK":
                        if kac_tane_listelendi < 5:
                            rapor_mesaji += f"🛑 *{sembol} LONG VETO!* (Ayı Piyasası Rejimi Aktif. Hisse senedi alımı yasak.)\n"
                        silinecek_adaylar.append(sembol)
                        is_adaylar_changed = True
                        continue
                    
                    # SEKTÖREL KORELASYON KALKANI
                    sektor_sayisi = 0
                    for port_sembol, port_veri in aktif_portfoy.items():
                        port_kimlik = VARLIK_DNA_DEPOSU.get(port_sembol, {})
                        if port_kimlik.get("Sektor", "") == kimlik["Sektor"]:
                            sektor_sayisi += 1
                            
                    if sektor_sayisi >= 2:
                        if kac_tane_listelendi < 5:
                            rapor_mesaji += f"🛡️ *{sembol} VETO!* ({kimlik['Sektor']} sektöründe limit dolu ({sektor_sayisi}/2))\n"
                        silinecek_adaylar.append(sembol)
                        is_adaylar_changed = True
                        continue

                    hesap = hesap_bilgisi_getir()
                    toplam_puan = guncel.get("Skor", 70) 

                    # ==========================================
                    # OPSIYON GEX MOTORU ENTEGRASYONU
                    # ==========================================
                    try:
                        from opsiyon_gex_motoru import karar_ve_risk_motoru
                        # Balina Radarı (Kısa Süreli Hafıza) kullanılarak Delta Eğilimi ve Fiyat Eğilimi içeride hesaplanır.
                        gex_karar = karar_ve_risk_motoru(
                            sembol=sembol, 
                            mevcut_kasa=hesap["Guncel_Bakiye"], 
                            ai_sinyali=islem_yonu, 
                            ai_guven=ai_olasiligi
                        )
                    except Exception as e:
                        print(f"[UYARI] {sembol} için Opsiyon GEX Motoru çalıştırılamadı: {e}")
                        gex_karar = None

                    # ==========================================
                    # KELLY KRİTERİ VE KASA YÖNETİMİ
                    # ==========================================
                    ai_basari_ihtimali = ai_olasiligi / 100.0 # Örn: 76.0 -> 0.76
                    
                    # Gerçekçi Eşik: Piyasada %51 devasa bir avantajdır.
                    if ai_basari_ihtimali < 0.51:
                        anlik_aksiyon = "PAS"
                        rapor_mesaji += f"⚠️ *{sembol} PAS GEÇİLDİ!* (Sinyal: {sinyal_sebebi} | Güven Yetersiz: %{ai_olasiligi:.1f})\n"
                        silinecek_adaylar.append(sembol)
                        is_adaylar_changed = True
                        continue
                        
                    # Kısmi Kelly (Fractional Kelly) Uyarlaması
                    # Çarpanı 0.10'dan 0.50'ye çıkardık.
                    # Örnek: %54 olasılık için -> (0.54 - 0.50) * 0.50 = 0.02 (Kasanın %2'si riske edilir)
                    dinamik_risk_yuzdesi = max(0.01, min(0.04, (ai_basari_ihtimali - 0.50) * 0.50))
                    
                    risk_profili_adi = f"META-SIZING (Sinyal: {sinyal_sebebi} | Güven: %{ai_olasiligi:.1f} | Risk: %{dinamik_risk_yuzdesi*100:.1f})"
                    
                    # Riske Edilecek Nakit Tutar
                    risk_tutari = hesap["Guncel_Bakiye"] * dinamik_risk_yuzdesi
                    
                    # Hisse Başına Alınan Risk (AFML'deki gibi 1.5 ATR)
                    risk_basina = abs(fiyat - stop) 
                    
                    # 4. PROFESYONEL LOT (ADET) HESAPLAMASI
                    adet = int(risk_tutari / risk_basina) if risk_basina > 0 else 1
                    
                    # ==========================================
                    # YENİ: DİNAMİK MAKSİMUM SERMAYE BAĞLAMA (DYNAMIC EXPOSURE CAP)
                    # ==========================================
                    # Kurumsal risk yönetimi: Sadece sabit bir limit yerine, Yapay Zeka'nın 
                    # sinyal güven oranına (ai_basari_ihtimali) göre esneyen bir tavan.
                    if ai_basari_ihtimali >= 0.80:
                        cap_orani = 0.25  # Çok yüksek güven: Kasanın %25'ine kadar izin
                    elif ai_basari_ihtimali >= 0.60:
                        cap_orani = 0.20  # Standart güven: Kasanın %20'si
                    else:
                        cap_orani = 0.10  # Sınırda güven (%51-60): Maksimum risk %10'a düşürülür
                        
                    maksimum_sermaye_limiti = hesap["Guncel_Bakiye"] * cap_orani
                    islem_maliyeti_tahmini = adet * fiyat
                    
                    if islem_maliyeti_tahmini > maksimum_sermaye_limiti:
                        # Eğer ATR hesabı bizi aşırı büyük bir pozisyona zorluyorsa, 
                        # lot miktarını "Dinamik Sermaye Tavanı" limitine göre tıraşla!
                        adet = int(maksimum_sermaye_limiti / fiyat)
                        risk_profili_adi += f" (Dinamik Sermaye Tavanı: %{cap_orani*100:.0f})"
                    
                    # GEX Motoru kararı varsa, adet ve riski güncelle
                    if gex_karar and "Hata" not in gex_karar:
                        adet = int(gex_karar.get("Lot", adet))
                        risk_profili_adi = gex_karar.get("Rejim", risk_profili_adi)
                        risk_tutari = gex_karar.get("Risk_Dolari", risk_tutari)
                        
                        if gex_karar.get("Aksiyon") == "PAS":
                            adet = 0 # İşlem GEX tarafından reddedildi
                            rapor_mesaji += f"⚠️ *{sembol} PAS GEÇİLDİ!* (GEX Motoru Gamma Rejimi Tespiti)\n"
                            print(f"[GEX RİSK YÖNETİMİ] {sembol} işlemi reddedildi. Rejim: {risk_profili_adi}")
                    
                    # ==========================================
                    # 5. İŞLEM MALİYETİ VE %30 NAKİT KALKANI (BUFFER)
                    # ==========================================
                    # Çift Katmanlı Rejim: Balon bölgesi lot küçültme (Risk Çarpanı)
                    adet = int(adet * risk_carpani)
                    if adet <= 0 and islem_yonu:
                        adet = 1
                        
                    birim_maliyet = fiyat * (1 + KAYMA_ORANI + KOMISYON_ORANI)
                    islem_maliyeti = adet * birim_maliyet
                    
                    # Kasanın her zaman %30'u "Dokunulmaz Acil Durum Fonu" olarak ayrılır.
                    # Örn: 10.000$ başlangıç kasasının 3.000$'ı asla işlemlere bağlanamaz.
                    minimum_nakit_siniri = hesap.get("Baslangic_Bakiyesi", 10000.0) * 0.30
                    kullanilabilir_nakit = hesap["Guncel_Bakiye"] - minimum_nakit_siniri
                    
                    if islem_maliyeti > kullanilabilir_nakit:
                        if kullanilabilir_nakit >= birim_maliyet:
                            # 1. Senaryo: Kotayı aşıyoruz ama biraz daha alım gücümüz var. (Lot sayısını küçült)
                            adet = int(kullanilabilir_nakit / birim_maliyet)
                            islem_maliyeti = adet * birim_maliyet
                            risk_profili_adi += " (%30 Nakit Kalkanı)"
                        else:
                            # 2. Senaryo: Kasa zaten %30 sınırına dayanmış. İşlemi acımasızca reddet.
                            adet = 0
                            print(f"[RİSK YÖNETİMİ] {sembol} işlemi reddedildi. Kasa %30 koruma sınırında!")
                            rapor_mesaji += f"⚠️ *{sembol} PAS GEÇİLDİ!* (%30 Nakit Kalkanı Devrede)\n"
                    
                    if adet > 0:
                        # Parayı kasadan düş
                        hesap["Guncel_Bakiye"] -= float(islem_maliyeti)
                        hesap_kaydet(hesap)
                        
                        guncel["Yon"] = islem_yonu
                        guncel["Stop"] = stop
                        guncel["Hedef"] = hedef
                        guncel["Adet"] = adet
                        guncel["Risk_Tutari"] = risk_tutari
                        guncel["Islem_Maliyeti"] = float(islem_maliyeti)
                        guncel["Risk_Profili"] = risk_profili_adi # Portföyde izlemek için kaydet
                        
                        # Telegram Mesajını Güncelle (Dinamik Risk Bilgisini Ekle)
                        foto_mesaj = (
                            f"[ALIM] *{sembol} SANAL {islem_yonu} ALIM YAPILDI!*\n\n"
                            f"*Sinyal Gücü:* {toplam_puan} Puan | AI: %{ai_olasiligi:.2f}\n"
                            f"*Trend Rejimi:* {rejim_mesaji}\n\n"
                            f"*Risk Profili:* {risk_profili_adi} (Kasanın %{dinamik_risk_yuzdesi*100:.1f}'i)\n"
                            f"*Giriş Fiyatı:* ${fiyat:.2f}\n"
                            f"*Stop:* ${stop:.2f} | *Hedef:* ${hedef:.2f}\n\n"
                            f"*Pozisyon:* {adet} Adet | *Maliyet:* ${islem_maliyeti:.2f}\n"
                            f"*(%0.1 Komisyon ve %0.05 Kayma dahil edilmiştir)*\n"
                            f"============================\n"
                            f"*Kalan Bakiye:* ${hesap['Guncel_Bakiye']:.2f}\n"
                            f"*Toplam P&L:* ${hesap['Toplam_Kar_Zarar']:.2f}\n"
                        )
                        
                        if guncel["Veri"] is not None and not guncel["Veri"].empty:
                            try:
                                grafik_dosyasi = grafik_ciz_ve_kaydet(guncel["Veri"], sembol, "AL" if islem_yonu == "LONG" else "SAT", fiyat, stop, hedef)
                                telegram_fotograf_gonder(grafik_dosyasi, foto_mesaj)
                            except: pass
                        
                        rapor_mesaji += f"🤖 *YAPAY ZEKA ONAYLI {islem_yonu} ALIM!* ({adet} Adet | Olasılık: %{ai_olasiligi})\n"
                        
                        aktif_portfoy[sembol] = guncel
                        silinecek_adaylar.append(sembol)
                        is_portfoy_changed = True
                        is_adaylar_changed = True
                        
                        # Anlık arayüz güncellemesi için portföyü hemen diske yaz (Gerçek Zamanlı UI)
                        dosya_yaz(PORTFOY_DOSYASI, aktif_portfoy)
                    else:
                        rapor_mesaji += f"❌ *{sembol} {islem_yonu} Alınamadı!* (AI %{ai_olasiligi} ama kasada yeterli bakiye yok. Bakiye: ${hesap['Guncel_Bakiye']:.2f})\n"
                else:
                    pass # İzlenenleri mesaja ekleme (spam engellendi)
            except Exception as e:
                print(f"[Hata] Aday havuzundaki {sembol} gun ici taranirken hata olustu: {e}")
        
        # Havuzdan çıkartılanları adaylar listesinden temizle
        for s in silinecek_adaylar:
            aday_havuzu.pop(s, None)

    # Dosyaları güncelle ve kaydet
    if is_adaylar_changed:
        dosya_yaz(ADAY_DOSYASI, aday_havuzu)
    if is_portfoy_changed:
        dosya_yaz(PORTFOY_DOSYASI, aktif_portfoy)
        
        # Ekstra: Mevcut Açık İşlemler ve P&L Özeti
        ozet_mesaji = rapor_mesaji + "\n\n📋 *MEVCUT AÇIK İŞLEMLER* 📋\n"
        ozet_mesaji += "-"*30 + "\n"
        if aktif_portfoy:
            toplam_portfoy_degeri = hesap["Guncel_Bakiye"]
            for s, v in aktif_portfoy.items():
                yon = v.get("Yon", "LONG")
                g_fiyat = v.get("Fiyat", 0)
                adet = v.get("Adet", 0)
                maliyet = v.get("Islem_Maliyeti", 0)
                stop = v.get("Stop", 0)
                hedef = v.get("Hedef", 0)
                ozet_mesaji += f"🔹 *{s}* ({yon}) - {adet} Adet\n"
                ozet_mesaji += f"   💵 Maliyet: ${maliyet:.2f} | Giriş: ${g_fiyat:.2f}\n"
                ozet_mesaji += f"   🛑 Stop: ${stop:.2f} | 🎯 Hedef: ${hedef:.2f}\n"
                toplam_portfoy_degeri += maliyet # (Yaklaşık değer)
            
            ozet_mesaji += "-"*30 + "\n"
            ozet_mesaji += f"💰 *Toplam Nakit Bakiye:* ${hesap['Guncel_Bakiye']:.2f}\n"
            ozet_mesaji += f"📈 *Gerçekleşen Toplam P&L:* ${hesap['Toplam_Kar_Zarar']:.2f}\n"
        else:
            ozet_mesaji += "Şu anda açık pozisyon bulunmuyor. Nakitte bekleniyor.\n"
            ozet_mesaji += f"💰 *Nakit Bakiye:* ${hesap['Guncel_Bakiye']:.2f}\n"
            ozet_mesaji += f"📈 *Gerçekleşen Toplam P&L:* ${hesap['Toplam_Kar_Zarar']:.2f}\n"

        send_telegram_message(ozet_mesaji)
        print("Portföy değişti, Telegram'a özet rapor iletildi.")
    else:
        print("Portföyde değişiklik yok, Telegram'a mesaj gönderilmedi (Sessiz Tarama).")


def is_us_market_open(us_saat):
    """ABD borsasının açık olup olmadığını kontrol eder (Pazartesi-Cuma, 09:30 - 16:00 EST)."""
    if us_saat.weekday() < 5:  # Pazartesi - Cuma
        market_dakika_baslangic = 9 * 60 + 30
        market_dakika_bitis = 16 * 60
        guncel_dakika = us_saat.hour * 60 + us_saat.minute
        if market_dakika_baslangic <= guncel_dakika < market_dakika_bitis:
            return True
    return False


if __name__ == "__main__":
    # Parametre girilmemişse varsayılan olarak CANLI TAKİP mod başlatılır
    if len(sys.argv) < 2:
        MOD = "LIVE"
    else:
        MOD = sys.argv[1].upper()
 
    print("=== MASTER BOT COK FAZLI TARAYICI BASLATILIYOR ===")
    print(f"Calisma Modu: {MOD}\n")

    # ÇEVRE İZOLASYONU: Canlı ve Sanal Dosyaları Ayır
    if MOD == "SANAL_TEST" or MOD == "ZAMAN_MAKINASI":
        PORTFOY_DOSYASI = "sanal_portfoy.json"
        HESAP_DOSYASI = "sanal_hesap.json"
        GECMIS_DOSYASI = "sanal_gecmis.csv"
    else:
        PORTFOY_DOSYASI = "canli_portfoy.json"
        HESAP_DOSYASI = "canli_hesap.json"
        GECMIS_DOSYASI = "canli_gecmis.csv"

    if MOD in ["LIVE", "CANLI", "CANLI_TAKIP"]:
        print("\n=== MASTER BOT CANLI TAKİP MODU BAŞLATILDI ===")
        print("Bot, ABD borsası açıkken her 15 dakikada bir intraday kontrol yapacak,")
        print("Market kapandığında ise günde 1 kez Gece Taraması çalıştıracaktır.\n")
        
        # Arka planda eksik verileri yama ve arşivleme işlemini başlat (Asenkron)
        try:
            import threading
            import veri_arsivcisi
            print("[SİSTEM] Veri Arşivleme Motoru arka planda başlatılıyor...")
            threading.Thread(target=veri_arsivcisi.guncelle, kwargs={"sessiz_mod": False}, daemon=True).start()
        except Exception as e:
            print(f"[UYARI] Veri arşivleme motoru başlatılamadı: {e}")
            
        gece_taramasi_tarihi = None
        
        # İlk açılışta bilgilendirme gönderelim
        baslangic_mesaji = (
            "🤖 *MASTER QUANT BOT AKTİF (CANLI TAKİP MODU)* 🤖\n\n"
            "Sistem kesintisiz çalışma modunda başlatıldı.\n"
            "📈 ABD Borsası açıkken 15 dakikalık periyotlarla takip yapılacak.\n"
            "🌙 Borsa kapandıktan sonra günlük Gece Taraması çalıştırılacak."
        )
        send_telegram_message(baslangic_mesaji)
        
        while True:
            try:
                us_saat = get_us_eastern_time()
                bugun_str = us_saat.strftime("%Y-%m-%d")
                market_acik = is_us_market_open(us_saat)
                
                print(f"\n[CANLI TAKİP] Sistem Zamanı (EST): {us_saat.strftime('%Y-%m-%d %H:%M:%S')} | Market: {'ACIK' if market_acik else 'KAPALI'}")
                
                if market_acik:
                    # Market açıkken gün içi kontrol yap
                    print("[CANLI TAKİP] Market açık. Gün içi kontrol çalıştırılıyor...")
                    calistir_gun_ici_kontrol()
                    
                    # 15 dakika bekle
                    print("[CANLI TAKİP] 15 dakika uyku moduna geçiliyor...")
                    time.sleep(15 * 60)
                else:
                    # Market kapalıyken
                    # Gece Taraması bugün henüz yapılmadıysa çalıştır
                    if gece_taramasi_tarihi != bugun_str:
                        print(f"[CANLI TAKİP] Market kapalı ve bugün için Gece Taraması henüz yapılmadı. Çalıştırılıyor...")
                        calistir_gece_taramasi()
                        
                        # Gece taramasından sonra gün sonu verilerini de arşive ekle
                        try:
                            import threading
                            import veri_arsivcisi
                            print("[SİSTEM] Gece Taraması bitti, Veri Arşivleme Motoru arka planda günlük snapshot alıyor...")
                            threading.Thread(target=veri_arsivcisi.guncelle, kwargs={"sessiz_mod": True}, daemon=True).start()
                        except: pass
                        
                        gece_taramasi_tarihi = bugun_str
                    
                    # Market kapalıyken 5 dakikada bir saati kontrol et
                    print("[CANLI TAKİP] Market kapalı. 5 dakika uyku moduna geçiliyor...")
                    time.sleep(5 * 60)
                    
            except KeyboardInterrupt:
                print("\n[CANLI TAKİP] Kullanıcı tarafından durduruldu (Ctrl+C). Çıkılıyor...")
                send_telegram_message("⚠️ *MASTER QUANT BOT DURDURULDU!* (Kullanıcı müdahalesi/Ctrl+C)")
                break
            except Exception as e:
                print(f"\n[CANLI TAKİP HATASI] Beklenmeyen hata oluştu: {e}")
                try:
                    send_telegram_message(f"🚨 *CANLI TAKİP KRİTİK HATASI:* {e}\n10 dakika sonra tekrar denenecek...")
                except:
                    pass
                time.sleep(10 * 60)

    elif MOD == "TEST_ALL":
        print("\n=== TEST_ALL MODU CALISTIRILIYOR ===")
        calistir_gece_taramasi()
        calistir_gun_ici_kontrol()
        print("\n=== TEST_ALL BASARIYLA TAMAMLANDI ===")
        
    elif MOD == "GECETARAMASI":
        calistir_gece_taramasi()
        
    elif MOD == "GUNICIKONTROL":
        calistir_gun_ici_kontrol()
        
    elif MOD == "SANAL_TEST":
        print("\n=== MASTER BOT SANAL TEST MODU BAŞLATILDI ===")
        print("Market kapalı olsa bile tüm portföy ve sinyaller (sanki market açıkmış gibi) test ediliyor...")
        calistir_gece_taramasi()
        while True:
            try:
                calistir_gun_ici_kontrol()
                print("[SANAL TEST] 15 dakika uyku moduna geçiliyor...")
                time.sleep(15 * 60)
            except KeyboardInterrupt:
                print("\n[SANAL TEST] Kullanıcı tarafından durduruldu (Ctrl+C). Çıkılıyor...")
                break
            except Exception as e:
                print(f"\n[SANAL TEST HATASI] Beklenmeyen hata oluştu: {e}")
                time.sleep(60)

    elif MOD == "ZAMAN_MAKINASI":
        print("\n=== 🚀 ZAMAN MAKİNESİ (WALK-FORWARD) BAŞLATILDI ===")
        print("Geçmiş 30 günün zaman çizelgesi çıkarılıyor, gelecek tamamen gizleniyor...")
        try:
            import yfinance as yf
            import pandas as pd
            spy_df = yf.download("SPY", period="30d", interval="15m", progress=False)
            if not spy_df.empty:
                zaman_cizelgesi = spy_df.index.tolist()
                print(f"Toplam {len(zaman_cizelgesi)} mumluk simülasyon başlıyor!")
                for t in zaman_cizelgesi:
                    SIMULASYON_ZAMANI = t.tz_localize(None) if hasattr(t, 'tz_localize') and t.tzinfo is not None else t
                    print(f"\n[ZAMAN MAKİNESİ] Sanal Saat İlerledi: {pd.to_datetime(SIMULASYON_ZAMANI).strftime('%Y-%m-%d %H:%M:%S')}")
                    calistir_gun_ici_kontrol()
                    # 1 saniye bekle (Gerçek hayatta 15 dakika)
                    time.sleep(1)
                print("\n=== 🚀 ZAMAN MAKİNESİ SİMÜLASYONU BİTTİ! ===")
            else:
                print("[HATA] Zaman çizelgesi oluşturulamadı!")
        except KeyboardInterrupt:
            print("\n[ZAMAN MAKİNESİ] Kullanıcı tarafından durduruldu.")
        except Exception as e:
            print(f"[HATA] Zaman Makinesi çöktü: {e}")
                
    else:
        print(f"[HATA] Bilinmeyen çalışma modu: {MOD}")
        print("Geçerli modlar: LIVE, TEST_ALL, GECETARAMASI, GUNICIKONTROL, SANAL_TEST, ZAMAN_MAKINASI")
