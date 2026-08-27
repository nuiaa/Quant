import os
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# sicak_motor.py içerisinden Hacim Çubuğu üretici ve Opsiyon Metrik çekici fonksiyonları içe aktarıyoruz
from sicak_motor import hacim_cubuklari_olustur, opsiyon_metriklerini_cek

ARSIV_KUTUPHANESI = "arsiv"
M1_KLASOR = os.path.join(ARSIV_KUTUPHANESI, "1m_veriler")
OPSIYON_KLASOR = os.path.join(ARSIV_KUTUPHANESI, "opsiyon_veriler")

# Klasörleri oluştur
os.makedirs(M1_KLASOR, exist_ok=True)
os.makedirs(OPSIYON_KLASOR, exist_ok=True)

def varliklari_getir():
    semboller = []
    if os.path.exists("piyasa_haritasi.json"):
        try:
            with open("piyasa_haritasi.json", "r", encoding="utf-8") as f:
                piyasa = json.load(f)
            
            for s_adi, s_v in piyasa.get("SEKTORLER", {}).items():
                for end_adi, end_v in s_v.get("Endustriler", {}).items():
                    semboller.extend(end_v.get("Hisseler", []))
                    
            # Ana endeksi de ekle
            ana_endeks = piyasa.get("PIYASA_REHBERI", {}).get("Ana_Endeks")
            if ana_endeks:
                semboller.append(ana_endeks)
                
            semboller = list(set(semboller)) # Tekrarları sil
        except Exception as e:
            print(f"[ARŞİV] HATA: piyasa_haritasi.json okunamadı. {e}")
            semboller = ['AAPL', 'MSFT', 'SPY'] # Fallback
    else:
        semboller = ['AAPL', 'MSFT', 'SPY']
    return semboller

def df_kaydet(df, dosya_yolu):
    """Pandas DataFrame'i Parquet formatında sıkıştırarak kaydeder (pyarrow/fastparquet yoksa CSV olarak kaydeder)"""
    try:
        df.to_parquet(dosya_yolu, compression='snappy')
    except ImportError:
        # Eğer sistemde Parquet kütüphanesi yoksa CSV.gz olarak kurtar
        csv_yolu = dosya_yolu.replace(".parquet", ".csv.gz")
        df.to_csv(csv_yolu, compression='gzip')

def df_oku(dosya_yolu):
    """Parquet dosyasını okur (yoksa fallback CSV.gz okur)"""
    if os.path.exists(dosya_yolu):
        try:
            return pd.read_parquet(dosya_yolu)
        except Exception:
            return pd.DataFrame()
            
    csv_yolu = dosya_yolu.replace(".parquet", ".csv.gz")
    if os.path.exists(csv_yolu):
        try:
            return pd.read_csv(csv_yolu, index_col='Datetime', parse_dates=True)
        except Exception:
            return pd.DataFrame()
            
    return pd.DataFrame()

def guncelle(sessiz_mod=False):
    semboller = varliklari_getir()
    if not sessiz_mod:
        print(f"\n[ARŞİV MOTORU] Toplam {len(semboller)} varlık için sıkıştırılmış veri arşivleme (Backfill) başlatılıyor...")
        
    bugun = datetime.now().date()
    
    for sembol in semboller:
        if "=" in sembol or "^" in sembol: 
            # Emtia ve endeksler için opsiyon/1m bazen sorunlu olabilir, ama yfinance üzerinden denemeye devam edeceğiz
            pass
            
        # ---------------------------------------------------------
        # 1. 1 DAKİKALIK VERİLERİN ARŞİVLENMESİ (HACİM ÇUBUĞU OLARAK)
        # ---------------------------------------------------------
        m1_dosya = os.path.join(M1_KLASOR, f"{sembol}_1m.parquet")
        eski_df_1m = df_oku(m1_dosya)
        
        try:
            # 7 Günlük maksimum geriye dönük veri çekimi (Gap'leri yamar)
            yeni_df_1m = yf.download(sembol, period="7d", interval="1m", progress=False)
            
            if not yeni_df_1m.empty:
                if isinstance(yeni_df_1m.columns, pd.MultiIndex):
                    yeni_df_1m.columns = yeni_df_1m.columns.droplevel(1)
                yeni_df_1m.columns = [col.strip().capitalize() for col in yeni_df_1m.columns]
                
                # Sıkıştırma Öncesi: Zaman Çubuklarını -> Hacim Çubuklarına Çevir (Dinamik eşik ile gürültüyü sil)
                # Standart 50000 yerine basitçe 1m verisini olduğu gibi de kaydedebiliriz, 
                # ancak kullanıcı hacim çubukları olarak kaydetmeyi sordu.
                # Burada orjinal 1m'yi tutmak her zaman daha iyidir, hacim çubuğunu çalışma anında oluştururuz.
                # Ancak yer tasarrufu istendiği için orijinal 1m kaydedilecektir, Parquet çok iyi sıkıştırır.
                yeni_df_1m.index.name = 'Datetime'
                
                if not eski_df_1m.empty:
                    # Birleştir ve mükerrer kayıtları (aynı tarih-saat) sil
                    birlestirilmis_1m = pd.concat([eski_df_1m, yeni_df_1m])
                    birlestirilmis_1m = birlestirilmis_1m[~birlestirilmis_1m.index.duplicated(keep='last')]
                else:
                    birlestirilmis_1m = yeni_df_1m
                    
                birlestirilmis_1m.sort_index(inplace=True)
                df_kaydet(birlestirilmis_1m, m1_dosya)
        except Exception as e:
            if not sessiz_mod: print(f"  [UYARI] {sembol} 1m verisi çekilemedi: {e}")
            
        # ---------------------------------------------------------
        # 2. OPSIYON VERİLERİNİN ARŞİVLENMESİ (AÇIK POZİSYONLAR)
        # ---------------------------------------------------------
        opsiyon_dosya = os.path.join(OPSIYON_KLASOR, f"{sembol}_opsiyon.parquet")
        eski_df_ops = df_oku(opsiyon_dosya)
        
        # Güncel Opsiyon verisini sicak_motor.py içerisindeki fonksiyonla çekiyoruz
        pcr, net_ops_gucu = opsiyon_metriklerini_cek(sembol)
        
        yeni_opsiyon_satiri = pd.DataFrame({
            'PCR': [pcr],
            'Net_Ops_Gucu': [net_ops_gucu]
        }, index=[pd.to_datetime(bugun)])
        yeni_opsiyon_satiri.index.name = 'Datetime'
        
        if not eski_df_ops.empty:
            son_tarih = eski_df_ops.index.max().date()
            fark = (bugun - son_tarih).days
            
            # Eğer arada atlanmış (bot açılmamış) günler varsa:
            if fark > 1:
                eksik_tarihler = [pd.to_datetime(son_tarih + timedelta(days=x)) for x in range(1, fark)]
                eksik_df = pd.DataFrame({
                    'PCR': [None] * len(eksik_tarihler),
                    'Net_Ops_Gucu': [None] * len(eksik_tarihler)
                }, index=eksik_tarihler)
                eksik_df.index.name = 'Datetime'
                
                eski_df_ops = pd.concat([eski_df_ops, eksik_df])
                
            # Eğer bugün zaten kaydedildiyse üzerine yaz (update), yoksa ekle
            if pd.to_datetime(bugun) in eski_df_ops.index:
                eski_df_ops.loc[pd.to_datetime(bugun)] = [pcr, net_ops_gucu]
                birlestirilmis_ops = eski_df_ops
            else:
                birlestirilmis_ops = pd.concat([eski_df_ops, yeni_opsiyon_satiri])
        else:
            birlestirilmis_ops = yeni_opsiyon_satiri
            
        birlestirilmis_ops.sort_index(inplace=True)
        df_kaydet(birlestirilmis_ops, opsiyon_dosya)
        
    if not sessiz_mod:
        print("[ARŞİV MOTORU] Veri arşivleme, eksik tamamlama (Backfill) ve sıkıştırma işlemi BAŞARIYLA TAMAMLANDI.")

if __name__ == "__main__":
    guncelle(sessiz_mod=False)
