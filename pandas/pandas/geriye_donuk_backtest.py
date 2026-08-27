import os
import sys
import time
import pickle
import sqlite3
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# sys.stdout encoding reconfiguration for absolute safe terminal handling
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Import mock or real neural architecture and scaler
from beyin_mimarisi import HibritQuantModeli
from sicak_motor import CustomMinMaxScaler, veriyi_oku_ve_ozellikleri_hesapla

print("==================================================")
print("=== HİBRİT MASTER BEYİN GERİYE DÖNÜK BACKTEST ===")
print("==================================================")

Cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BILGI] Backtest Cihazi: {Cihaz}")

pd.options.mode.chained_assignment = None

def yon_izni_ve_reversion_filtresi(spot_fiyat, ema200, ema20, rsi_degeri, sapma_esigi=0.20):
    uzaklik_orani = (spot_fiyat - ema200) / ema200 if ema200 > 0 else 0
    izinler = {"LONG": False, "SHORT": False, "Rejim": "", "Risk_Carpani": 1.0}

    if spot_fiyat > ema200:
        izinler["LONG"] = True 
        if uzaklik_orani >= sapma_esigi:
            izinler["Risk_Carpani"] = 0.5 
            if spot_fiyat < ema20 or rsi_degeri < 70:
                izinler["SHORT"] = True
                izinler["Rejim"] = f"AŞIRI ALIM (+%{uzaklik_orani*100:.1f}) | Momentum Kırıldı -> SHORT İzni AKTİF"
            else:
                izinler["Rejim"] = f"AŞIRI ALIM (+%{uzaklik_orani*100:.1f}) | Ralli Devam Ediyor -> SHORT VETO (Sadece %50 Riskli LONG)"
        else:
            izinler["Rejim"] = "NORMAL BOĞA | Sadece LONG Serbest"
    else:
        izinler["SHORT"] = True 
        if uzaklik_orani <= -sapma_esigi:
            izinler["Risk_Carpani"] = 0.5
            if spot_fiyat > ema20 or rsi_degeri > 30:
                izinler["LONG"] = True
                izinler["Rejim"] = f"AŞIRI SATIM (%{uzaklik_orani*100:.1f}) | Dönüş Teyit Edildi -> LONG İzni AKTİF"
            else:
                izinler["Rejim"] = f"AŞIRI SATIM (%{uzaklik_orani*100:.1f}) | Şelale Sürüyor -> LONG VETO (Sadece %50 Riskli SHORT)"
        else:
            izinler["Rejim"] = "NORMAL AYI | Sadece SHORT Serbest"
            
    return izinler

def master_ai_karar_motoru(prob_long, prob_short, spot_fiyat, ema200, adx_degeri, hisse_sektoru, makro_spy_trend, esik_long=55.0):
    fark_long_lehine = prob_long - prob_short
    fark_short_lehine = prob_short - prob_long
    boga_piyasasi = spot_fiyat > ema200
    
    is_tech = hisse_sektoru.upper() in ["TEKNOLOJİ", "YAZILIM", "YARI İLETKEN", "TEKNOLOJI", "BILISIM"]
    if makro_spy_trend == "BOGA" and is_tech:
        short_yasak = True
    else:
        short_yasak = False

    # LONG KARARI (Asimetrik Zeka + ADX)
    if prob_long >= esik_long and fark_long_lehine >= 10.0:
        if boga_piyasasi and adx_degeri >= 25.0:
            return "LONG", "GÜÇLÜ TREND: ADX Onaylı Asimetrik LONG (Rokete Binildi)"
        elif boga_piyasasi and adx_degeri < 25.0:
            return "PAS", "VETO: AI Long diyor ama ADX zayıf (Yatay piyasa gürültüsü)"
        else:
            return "LONG", "DİPTEN DÖNÜŞ: Ayı piyasasında tepki alımı yakalandı."

    # SHORT KARARI (Katı İspat Zorunluluğu)
    if prob_short >= 75.0 and fark_short_lehine >= 35.0:
        if short_yasak:
            return "PAS", "VETO: Teknoloji hissesinde Boğa piyasasında SHORT açılamaz! İntihar engellendi."
        return "SHORT", "NET ÇÖKÜŞ: Yüksek güvenli ve geniş farkla onaylanmış SHORT."

    return "PAS", "KARARSIZ: AI olasılıkları tatmin edici spread yaratamadı."

def short_borrow_maliyeti_hesapla(islem_tutari_dolari, elde_tutulan_gun, yillik_borrow_rate=0.08):
    # Backtest için direkt gün bazlı hesaplıyoruz (çünkü backtest günlük iterasyon yapıyor)
    elde_tutulan_gun_gercek = max(1.0, float(elde_tutulan_gun))
    gunluk_faiz_orani = yillik_borrow_rate / 365.0
    kiralama_maliyeti_dolari = islem_tutari_dolari * gunluk_faiz_orani * elde_tutulan_gun_gercek
    return round(kiralama_maliyeti_dolari, 2)

def dinamik_kademeli_trailing_stop(pozisyon_yonu, giris_fiyati, anlik_fiyat, mevcut_stop, atr_degeri):
    if pozisyon_yonu == "LONG":
        kar_miktari = anlik_fiyat - giris_fiyati
        kar_atr_cinsinden = kar_miktari / atr_degeri if atr_degeri > 0 else 0
    else: # SHORT
        kar_miktari = giris_fiyati - anlik_fiyat
        kar_atr_cinsinden = kar_miktari / atr_degeri if atr_degeri > 0 else 0
        
    if kar_atr_cinsinden >= 1.5:
        carpan = 1.0
    elif kar_atr_cinsinden >= 1.0:
        carpan = 1.5
    else:
        carpan = 1.5
        
    yeni_stop = mevcut_stop
    if pozisyon_yonu == "LONG":
        hesaplanan_yeni_stop = anlik_fiyat - (atr_degeri * carpan)
        if hesaplanan_yeni_stop > mevcut_stop:
            yeni_stop = hesaplanan_yeni_stop
    elif pozisyon_yonu == "SHORT":
        hesaplanan_yeni_stop = anlik_fiyat + (atr_degeri * carpan)
        if hesaplanan_yeni_stop < mevcut_stop:
            yeni_stop = hesaplanan_yeni_stop
    return round(yeni_stop, 2), carpan

# 1. MODEL VE SCALER ÖNBELLEĞİ (Sektörel Dinamik Yükleme)
from beyin_mimarisi import DinamikHiyerarsikModel
import json

SEKTOR_HARITASI = {}
if os.path.exists("piyasa_haritasi.json"):
    try:
        with open("piyasa_haritasi.json", "r", encoding="utf-8") as f:
            piyasa = json.load(f)
        for s_adi, s_v in piyasa.get("SEKTORLER", {}).items():
            for end_adi, end_v in s_v.get("Endustriler", {}).items():
                for hisse in end_v.get("Hisseler", []):
                    SEKTOR_HARITASI[hisse] = s_adi
    except Exception:
        pass

# Modelleri ve scaler'ları hafızada tutacağız
MODEL_CACHE = {}

def get_sektor_modeli(sembol):
    sektor_adi = SEKTOR_HARITASI.get(sembol, "TEKNOLOJI") # Default Teknoloji
    if sembol in ['GC=F', 'SI=F']:
        sektor_adi = 'EMTIA_VE_MADENCILIK'
        
    hedef_model_adi = sektor_adi.replace(" ", "_").upper()
    
    if hedef_model_adi in MODEL_CACHE:
        return MODEL_CACHE[hedef_model_adi]
        
    model_long_dosyasi = f"{hedef_model_adi}_long_hiyerarsik_beyin.pth"
    model_short_dosyasi = f"{hedef_model_adi}_short_hiyerarsik_beyin.pth"
    scaler_m_dosyasi = f"{hedef_model_adi}_scaler_makro.pkl"
    scaler_t_dosyasi = f"{hedef_model_adi}_scaler_teknik.pkl"
    
    if not os.path.exists(model_long_dosyasi):
        # Fallback to TEKNOLOJI if specific sector model not found
        hedef_model_adi = "TEKNOLOJI"
        model_long_dosyasi = f"{hedef_model_adi}_long_hiyerarsik_beyin.pth"
        model_short_dosyasi = f"{hedef_model_adi}_short_hiyerarsik_beyin.pth"
        scaler_m_dosyasi = f"{hedef_model_adi}_scaler_makro.pkl"
        scaler_t_dosyasi = f"{hedef_model_adi}_scaler_teknik.pkl"

    model_long = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)
    if os.path.exists(model_long_dosyasi):
        model_long.load_state_dict(torch.load(model_long_dosyasi, map_location=Cihaz))
    model_long.eval()

    model_short = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)
    if os.path.exists(model_short_dosyasi):
        model_short.load_state_dict(torch.load(model_short_dosyasi, map_location=Cihaz))
    model_short.eval()

    try:
        with open(scaler_m_dosyasi, "rb") as f: scaler_m = pickle.load(f)
        with open(scaler_t_dosyasi, "rb") as f: scaler_t = pickle.load(f)
    except:
        scaler_m, scaler_t = None, None
        
    MODEL_CACHE[hedef_model_adi] = (model_long, model_short, scaler_m, scaler_t)
    print(f"[SISTEM] {hedef_model_adi} Sektör Beyinleri yüklendi.")
    return MODEL_CACHE[hedef_model_adi]

print("[SISTEM] Dinamik Sektörel Yükleme Altyapısı Hazır!")

# 2. VERİLERİ DATABASE'DEN ÇEK VE HAZIRLA
SEMBOL_LISTESI = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK_B']
dfs = {}
print("\n[VERI] Veritabanından tarihsel hisse verileri yükleniyor...")
for sembol in SEMBOL_LISTESI:
    try:
        df = veriyi_oku_ve_ozellikleri_hesapla(sembol)
        if df is not None and len(df) > 100:
            df.index = pd.to_datetime(df.index)
            dfs[sembol] = df
            print(f"   {sembol:6} | Satır sayısı: {len(df):4} | Veri Aralığı: {df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")
        else:
            print(f"   [UYARI] {sembol} için yetersiz veri, geçiliyor.")
    except Exception as e:
        print(f"   [HATA] {sembol} verisi yüklenirken hata oluştu: {e}")

if not dfs:
    print("[HATA] Yüklenecek veri bulunamadı!")
    sys.exit(1)

# 3. YAZILIM ENTEGRASYONU VE GEÇMİŞE DÖNÜK YAPAY ZEKA TAHMİNLERİ
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

predict_data = {}
print("\n[AI] Tarihsel Yapay Zeka çıkarımları (Inference) hesaplanıyor...")

for sembol, df in dfs.items():
    t_basla = time.time()
    
    # Sektöre özel modeli ve scaleri al
    model_long, model_short, scaler_m, scaler_t = get_sektor_modeli(sembol)
    
    # NaN temizliği ve özellikleri çekme
    for col in makro_ozellikler + teknik_ozellikler:
        if col not in df.columns:
            df[col] = 0.0
            
    X_m_ham = df[makro_ozellikler].values
    X_t_ham = df[teknik_ozellikler].values
    
    if scaler_m and scaler_t:
        X_m_olcekli = scaler_m.transform(X_m_ham)
        X_t_olcekli = scaler_t.transform(X_t_ham)
    else:
        X_m_olcekli = X_m_ham
        X_t_olcekli = X_t_ham
    
    # ATR bazlı dinamik Stop-Loss ve Take-Profit (Direnç Hedefi)
    df['Stop_Loss'] = df['Close'] - (df['ATR'] * 2.0)
    dinamik_hedef = df['Close'] + (df['ATR'] * 3.0)
    klasik_direnc = df['High'].rolling(window=20).max()
    df['Direnc_Hedefi'] = pd.concat([dinamik_hedef, klasik_direnc], axis=1).max(axis=1)
    
    # 60 günlük pencereleri modelden geçirerek Yapay Zeka olasılıklarını doldur
    ai_probs_long = np.zeros(len(df))
    ai_probs_short = np.zeros(len(df))
    N = len(df)
    
    # Pencereleri sequence haline getirip prediction yapmak
    m_pencereler = []
    t_pencereler = []
    indeksler = []
    for idx in range(60, N):
        t_pencereler.append(X_t_olcekli[idx-60:idx])
        m_pencereler.append(X_m_olcekli[idx-1]) # O günün makrosu
        indeksler.append(idx)
        
        if len(t_pencereler) >= 1000:
            b_m = torch.tensor(np.array(m_pencereler), dtype=torch.float32).to(Cihaz)
            b_t = torch.tensor(np.array(t_pencereler), dtype=torch.float32).to(Cihaz)
            with torch.no_grad():
                outputs_long = torch.sigmoid(model_long(b_m, b_t)).cpu().numpy().flatten()
                outputs_short = torch.sigmoid(model_short(b_m, b_t)).cpu().numpy().flatten()
            for i, val_l, val_s in zip(indeksler, outputs_long, outputs_short):
                ai_probs_long[i] = val_l
                ai_probs_short[i] = val_s
            m_pencereler, t_pencereler, indeksler = [], [], []
            
    if t_pencereler:
        b_m = torch.tensor(np.array(m_pencereler), dtype=torch.float32).to(Cihaz)
        b_t = torch.tensor(np.array(t_pencereler), dtype=torch.float32).to(Cihaz)
        with torch.no_grad():
            outputs_long = torch.sigmoid(model_long(b_m, b_t)).cpu().numpy().flatten()
            outputs_short = torch.sigmoid(model_short(b_m, b_t)).cpu().numpy().flatten()
        for i, val_l, val_s in zip(indeksler, outputs_long, outputs_short):
            ai_probs_long[i] = val_l
            ai_probs_short[i] = val_s
            
    df['AI_Olasiligi_Long'] = ai_probs_long * 100.0
    df['AI_Olasiligi_Short'] = ai_probs_short * 100.0
    df['AI_Olasiligi'] = df['AI_Olasiligi_Long'] # Geriye dönük uyumluluk için
    
    # Simülasyonun 10 yıllık (Kapsamlı) çalışması için 2016-01-01 sonrasını filtreleyelim
    df_filtered = df.loc[df.index >= '2016-01-01'].copy()
    predict_data[sembol] = df_filtered
    sure_sn = time.time() - t_basla
    
    probs_2016 = df_filtered['AI_Olasiligi_Long'].values
    p_min, p_max, p_mean = probs_2016.min(), probs_2016.max(), probs_2016.mean()
    c_50 = np.sum(probs_2016 >= 50.0)
    c_60 = np.sum(probs_2016 >= 60.0)
    c_70 = np.sum(probs_2016 >= 70.0)
    c_80 = np.sum(probs_2016 >= 80.0)
    print(f"   {sembol:6} | Çıkarım tamamlandı. Süre: {sure_sn:.2f} sn | 2016+ Gün: {len(df_filtered)} | Min: {p_min:.1f}%, Max: {p_max:.1f}%, Ort: {p_mean:.1f}%")
    print(f"            | Olasılık Dağılımı (Long): >=50%: {c_50:3} | >=60%: {c_60:3} | >=70%: {c_70:3} | >=80%: {c_80:3}")

# 4. KRONOLOJİK SİMÜLASYON FONKSİYONU
all_dates = sorted(list(set().union(*[predict_data[s].index for s in predict_data])))
baslangic_bakiye = 10000.0
risk_orani = 0.08

def simulasyon_calistir(al_esigi, sat_esigi):
    bakiye = baslangic_bakiye
    nakit = baslangic_bakiye
    aktif_pozisyonlar = []
    islem_gecmisi = []
    
    for gun in all_dates:
        # A. AKTİF POZİSYONLARIN DURUMUNU GÜNCELLE VE ÇIKIŞLARI KONTROL ET
        guncel_pozisyonlar = []
        for pos in aktif_pozisyonlar:
            sembol = pos['sembol']
            df_sembol = predict_data[sembol]
            
            if gun not in df_sembol.index:
                guncel_pozisyonlar.append(pos)
                continue
                
            # Dinamik Esik Belirleme
            if al_esigi == 52.0:
                esik_limit = 55.0 # Optimize modunda Long giriş eşiği 55.0'a çekildi (Agresif Büyüme)
                cikis_esik_limit = 40.0 # Erken çıkış eşiği 40.0'a çekildi
            else:
                esik_limit = al_esigi
                cikis_esik_limit = sat_esigi
                
            row = df_sembol.loc[gun]
            high = float(row['High'])
            low = float(row['Low'])
            close = float(row['Close'])
            open_val = float(row['Open'])
            ai_prob_long = float(row['AI_Olasiligi_Long'])
            ai_prob_short = float(row['AI_Olasiligi_Short'])
            
            pos['days_held'] += 1
            cikis_yapildi = False
            cikis_fiyati = 0.0
            sebep = ""
            
            if pos.get('yon', 'LONG') == 'LONG':
                pos['max_high'] = max(pos['max_high'], high)
                
                # 1. TAKE-PROFIT (UPPER BARRIER)
                hedef_fiyati = pos['entry_price'] + (pos['entry_atr'] * 1.5)
                if high >= hedef_fiyati:
                    cikis_fiyati = hedef_fiyati
                    if open_val > hedef_fiyati:
                        cikis_fiyati = open_val
                    sebep = "TAKE-PROFIT (Hedef Vuruldu)"
                    cikis_yapildi = True
                    
                # 2. STOP-LOSS (LOWER BARRIER)
                elif low <= pos['stop_loss']:
                    cikis_fiyati = pos['stop_loss']
                    if open_val < pos['stop_loss']:
                        cikis_fiyati = open_val
                    sebep = "STOP-LOSS"
                    cikis_yapildi = True
                # 2. GÜÇLÜ SHORT SİNYALİ (ERKEN REVERSAL)
                elif ai_prob_short >= 75.0:
                    cikis_fiyati = close
                    sebep = f"AI REVERSAL SİNYALİ (SHORT %{ai_prob_short:.1f})"
                    cikis_yapildi = True
                # 3. KADEMELİ DİNAMİK İZLEYEN STOP
                if not cikis_yapildi:
                    eski_stop = pos['stop_loss']
                    yeni_stop, _ = dinamik_kademeli_trailing_stop("LONG", pos['entry_price'], close, eski_stop, pos['entry_atr'])
                    if yeni_stop != eski_stop:
                        pos['stop_loss'] = yeni_stop
                        if low <= pos['stop_loss']:
                            cikis_fiyati = pos['stop_loss']
                            if open_val < pos['stop_loss']:
                                cikis_fiyati = open_val
                            sebep = "KARLA_STOP_OLDU (Step-Up Trailing Stop)"
                            cikis_yapildi = True
                
                # 4. TIME BARRIER (ZAMANAŞIMI)
                if not cikis_yapildi and pos['days_held'] >= 15:
                    cikis_fiyati = close
                    sebep = "TIME BARRIER (15 Gün Zaman Aşımı)"
                    cikis_yapildi = True
                                
            else: # SHORT POZİSYON KONTROLLLERİ
                pos['min_low'] = min(pos['min_low'], low)
                
                # 1. TAKE-PROFIT SHORT (LOWER BARRIER)
                hedef_fiyati = pos['entry_price'] - (pos['entry_atr'] * 1.5)
                if low <= hedef_fiyati:
                    cikis_fiyati = hedef_fiyati
                    if open_val < hedef_fiyati:
                        cikis_fiyati = open_val
                    sebep = "TAKE-PROFIT SHORT (Hedef Vuruldu)"
                    cikis_yapildi = True
                    
                # 2. STOP-LOSS SHORT (UPPER BARRIER)
                elif high >= pos['stop_loss']:
                    cikis_fiyati = pos['stop_loss']
                    if open_val > pos['stop_loss']:
                        cikis_fiyati = open_val
                    sebep = "STOP-LOSS SHORT"
                    cikis_yapildi = True
                # 2. GÜÇLÜ LONG SİNYALİ (ERKEN REVERSAL)
                elif ai_prob_long >= esik_limit:
                    cikis_fiyati = close
                    sebep = f"AI REVERSAL SİNYALİ (LONG %{ai_prob_long:.1f})"
                    cikis_yapildi = True
                # 3. KADEMELİ DİNAMİK İZLEYEN STOP (SHORT İÇİN)
                if not cikis_yapildi:
                    eski_stop = pos['stop_loss']
                    yeni_stop, _ = dinamik_kademeli_trailing_stop("SHORT", pos['entry_price'], close, eski_stop, pos['entry_atr'])
                    if yeni_stop != eski_stop:
                        pos['stop_loss'] = yeni_stop
                        if high >= pos['stop_loss']:
                            cikis_fiyati = pos['stop_loss']
                            if open_val > pos['stop_loss']:
                                cikis_fiyati = open_val
                            sebep = "KARLA_STOP_OLDU SHORT (Step-Up Trailing Stop)"
                            cikis_yapildi = True
                            
                # 4. TIME BARRIER (ZAMANAŞIMI)
                if not cikis_yapildi and pos['days_held'] >= 15:
                    cikis_fiyati = close
                    sebep = "TIME BARRIER (15 Gün Zaman Aşımı)"
                    cikis_yapildi = True
                                
            if cikis_yapildi:
                if pos.get('yon', 'LONG') == 'LONG':
                    tutar = pos['qty'] * cikis_fiyati
                    nakit += tutar
                    pl = tutar - (pos['qty'] * pos['entry_price'])
                else: # SHORT
                    pl = (pos['entry_price'] - cikis_fiyati) * pos['qty']
                    borrow_maliyeti = short_borrow_maliyeti_hesapla(pos['qty'] * pos['entry_price'], pos['days_held'])
                    pl -= borrow_maliyeti
                    tutar = (pos['qty'] * pos['entry_price']) + pl
                    nakit += tutar
                
                pl_pct = pl / (pos['qty'] * pos['entry_price']) * 100.0
                
                islem_gecmisi.append({
                    'Sembol': sembol,
                    'Yon': pos.get('yon', 'LONG'),
                    'Giris_Tarihi': pos['entry_date'].strftime('%Y-%m-%d'),
                    'Giris_Fiyati': pos['entry_price'],
                    'Cikis_Tarihi': gun.strftime('%Y-%m-%d'),
                    'Cikis_Fiyati': cikis_fiyati,
                    'Adet': pos['qty'],
                    'Hacim_USD': pos['qty'] * pos['entry_price'],
                    'P&L_USD': pl,
                    'P&L_Pct': pl_pct,
                    'Sebep': sebep,
                    'Elde_Tutma_Gun': pos['days_held']
                })
            else:
                guncel_pozisyonlar.append(pos)
                
        aktif_pozisyonlar = guncel_pozisyonlar
        
        # B. KASA PORTFÖY DEĞERİNİ HESAPLA
        portfoy_degeri = nakit
        for pos in aktif_pozisyonlar:
            sembol = pos['sembol']
            df_sembol = predict_data[sembol]
            c_price = float(df_sembol.loc[gun]['Close']) if gun in df_sembol.index else pos['entry_price']
            
            if pos.get('yon', 'LONG') == 'LONG':
                portfoy_degeri += pos['qty'] * c_price
            else: # SHORT
                # Value = Allocated margin + current P&L
                pl_now = (pos['entry_price'] - c_price) * pos['qty']
                portfoy_degeri += (pos['qty'] * pos['entry_price']) + pl_now
                
        bakiye = portfoy_degeri
        
        # C. ALIM SİNYALLERİNİ YAKALA VE İŞLEME GİR (LONG VEYA SHORT)
        adaylar = []
        for sembol in predict_data:
            df_sembol = predict_data[sembol]
            if gun not in df_sembol.index:
                continue
            if any(pos['sembol'] == sembol for pos in aktif_pozisyonlar):
                continue
                
            row = df_sembol.loc[gun]
            ai_prob_long = float(row['AI_Olasiligi_Long'])
            ai_prob_short = float(row['AI_Olasiligi_Short'])
            
            spot_fiyat = float(row['Close'])
            ema200 = float(row.get('EMA_200', spot_fiyat))
            ema20 = float(row.get('EMA_20', spot_fiyat))
            rsi_degeri = float(row.get('RSI', 50.0))
            adx_degeri = float(row.get('ADX', 25.0))
            
            izinler = yon_izni_ve_reversion_filtresi(spot_fiyat, ema200, ema20, rsi_degeri, sapma_esigi=0.20)
            
            # Dinamik Esik Belirleme
            if al_esigi == 52.0:
                esik_limit = 55.0 # Optimize modunda Long giriş eşiği 55.0'a çekildi (Agresif Büyüme)
            else:
                esik_limit = al_esigi

            # ==========================================
            # PRIMARY MODEL: AI NE DİYOR? (Saf Olasılıklar)
            # ==========================================
            if ai_prob_long >= ai_prob_short:
                ai_karari = "LONG"
            else:
                ai_karari = "SHORT"
                
            if ai_karari == "LONG" and izinler["LONG"]:
                adaylar.append({'sembol': sembol, 'yon': 'LONG', 'row': row, 'ai_prob': ai_prob_long, 'risk_carpani': izinler.get("Risk_Carpani", 1.0)})
            elif ai_karari == "SHORT" and izinler["SHORT"]:
                adaylar.append({'sembol': sembol, 'yon': 'SHORT', 'row': row, 'ai_prob': ai_prob_short, 'risk_carpani': izinler.get("Risk_Carpani", 1.0)})

                
        adaylar = sorted(adaylar, key=lambda x: x['ai_prob'], reverse=True)
        for aday in adaylar:
            sembol = aday['sembol']
            yon = aday['yon']
            row = aday['row']
            close_price = float(row['Close'])
            atr = float(row['ATR'])
            
            # Üçlü Bariyer parametrelerine göre Stop belirleme (AFML: 1.5 ATR)
            if yon == 'LONG':
                stop_loss = close_price - (atr * 1.5)
            else: # SHORT
                stop_loss = close_price + (atr * 1.5)
            
            # ==========================================
            # DİNAMİK KASA YÖNETİMİ (AFML META-LABELING SIZING)
            # ==========================================
            # Modelin verdiği "Başarı İhtimali" (% olarak) ne kadar yüksekse, o kadar risk al.
            ai_prob = aday['ai_prob']
            ai_basari_ihtimali = ai_prob / 100.0 # Örn: 76.0 -> 0.76
            
            # Gerçekçi Eşik: Piyasada %51 devasa bir avantajdır.
            if ai_basari_ihtimali < 0.51:
                continue
                
            # Kısmi Kelly (Fractional Kelly) Uyarlaması
            # Çarpanı 0.10'dan 0.50'ye çıkardık.
            # Örnek: %54 olasılık için -> (0.54 - 0.50) * 0.50 = 0.02 (Kasanın %2'si riske edilir)
            dinamik_risk_yuzdesi = max(0.01, min(0.04, (ai_basari_ihtimali - 0.50) * 0.50))
            
            # Riske Edilecek Nakit Tutar
            riske_edilen_dolar = bakiye * dinamik_risk_yuzdesi
            
            # Hisse Başına Alınan Risk (AFML'deki gibi 1.5 ATR)
            hisse_basi_risk_dolar = atr * 1.5
            
            if hisse_basi_risk_dolar <= 0:
                continue
                
            adet = int(riske_edilen_dolar / hisse_basi_risk_dolar)
            if adet <= 0:
                adet = 1
                
            # Kaldıraçsız işlem (Kasadan fazla alınamaz)
            alinacak_tutar = adet * close_price
            if alinacak_tutar > nakit:
                adet = int(nakit / close_price)
                
            gerekli_para = adet * close_price
                
            if adet > 0 and gerekli_para <= nakit:
                nakit -= gerekli_para
                aktif_pozisyonlar.append({
                    'sembol': sembol,
                    'yon': yon,
                    'entry_date': gun,
                    'entry_price': close_price,
                    'stop_loss': stop_loss,
                    'qty': adet,
                    'entry_atr': atr,
                    'max_high': close_price,
                    'min_low': close_price,
                    'days_held': 0
                })
                
    # AÇIK POZİSYONLARI SON GÜN KAPAT
    son_tarih = all_dates[-1]
    for pos in aktif_pozisyonlar:
        sembol = pos['sembol']
        df_sembol = predict_data[sembol]
        cikis_fiyati = float(df_sembol.loc[son_tarih]['Close']) if son_tarih in df_sembol.index else pos['entry_price']
        
        if pos.get('yon', 'LONG') == 'LONG':
            tutar = pos['qty'] * cikis_fiyati
            nakit += tutar
            pl = tutar - (pos['qty'] * pos['entry_price'])
        else: # SHORT
            pl = (pos['entry_price'] - cikis_fiyati) * pos['qty']
            borrow_maliyeti = short_borrow_maliyeti_hesapla(pos['qty'] * pos['entry_price'], pos['days_held'])
            pl -= borrow_maliyeti
            tutar = (pos['qty'] * pos['entry_price']) + pl
            nakit += tutar
            
        pl_pct = pl / (pos['qty'] * pos['entry_price']) * 100.0
        
        islem_gecmisi.append({
            'Sembol': sembol,
            'Yon': pos.get('yon', 'LONG'),
            'Giris_Tarihi': pos['entry_date'].strftime('%Y-%m-%d'),
            'Giris_Fiyati': pos['entry_price'],
            'Cikis_Tarihi': son_tarih.strftime('%Y-%m-%d'),
            'Cikis_Fiyati': cikis_fiyati,
            'Adet': pos['qty'],
            'Hacim_USD': pos['qty'] * pos['entry_price'],
            'P&L_USD': pl,
            'P&L_Pct': pl_pct,
            'Sebep': 'AÇIK POZİSYON SONLANDIRILDI',
            'Elde_Tutma_Gun': pos['days_held']
        })
        
    trades_df = pd.DataFrame(islem_gecmisi)
    if trades_df.empty:
        expected_cols = [
            'Sembol', 'Yon', 'Giris_Tarihi', 'Giris_Fiyati', 'Cikis_Tarihi', 
            'Cikis_Fiyati', 'Adet', 'Hacim_USD', 'P&L_USD', 'P&L_Pct', 'Sebep', 'Elde_Tutma_Gun'
        ]
        trades_df = pd.DataFrame(columns=expected_cols)
        
    return nakit, trades_df

# SİMÜLASYONLARI ÇALIŞTIR
print(f"\n[SİMÜLASYON] 1. Model Çalıştırılıyor: Orijinal Sıkı Mod (%80.0 Alım / %45.0 Çıkış)...")
bakiye_strict, trades_strict = simulasyon_calistir(80.0, 45.0)

print(f"[SİMÜLASYON] 2. Model Çalıştırılıyor: Calibre Edilmiş Optimize Mod (S&P 500: %58.0, XAUUSD: %65.0 Alım / %50.5 Çıkış)...")
bakiye_calib, trades_calib = simulasyon_calistir(52.0, 50.5)

# 5. BUY & HOLD (ENDEKS) BENCHMARK HESAPLAMA
buy_hold_final = 0.0
start_date = all_dates[0]
end_date = all_dates[-1]

print("\n[BENCHMARK] Buy & Hold (Eşit Ağırlıklı Endeks) karşılaştırması hesaplanıyor...")
for sembol in SEMBOL_LISTESI:
    df_sembol = predict_data[sembol]
    baslangic_fiyat = float(df_sembol.iloc[0]['Close'])
    son_fiyat = float(df_sembol.iloc[-1]['Close'])
    
    hisse_bakiye = 1000.0
    hisse_adedi = hisse_bakiye / baslangic_fiyat
    hisse_final_deger = hisse_adedi * son_fiyat
    buy_hold_final += hisse_final_deger
    
    print(f"   {sembol:6} | Giriş: ${baslangic_fiyat:6.2f} | Çıkış: ${son_fiyat:6.2f} | Getiri: %{((son_fiyat - baslangic_fiyat)/baslangic_fiyat*100):+7.2f}")

buy_hold_getiri_pct = (buy_hold_final - baslangic_bakiye) / baslangic_bakiye * 100.0

# Calibre edilmiş detayları kaydet
trades_calib.to_csv("geriye_donuk_backtest_detayli.csv", index=False, encoding='utf-8-sig')

# İstatistik Hesaplama Fonksiyonu
def metrikleri_yazdir(bakiye_son, df_trades, etiket):
    total_trades = len(df_trades)
    wins = len(df_trades[df_trades['P&L_USD'] >= 0]) if total_trades > 0 else 0
    losses = len(df_trades[df_trades['P&L_USD'] < 0]) if total_trades > 0 else 0
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    
    model_net_kar = bakiye_son - baslangic_bakiye
    model_getiri_pct = model_net_kar / baslangic_bakiye * 100.0
    
    if total_trades > 0:
        trades_sorted = df_trades.sort_values(by='Cikis_Tarihi')
        cur_bal = baslangic_bakiye
        bakiyeler = [baslangic_bakiye]
        for _, t in trades_sorted.iterrows():
            cur_bal += t['P&L_USD']
            bakiyeler.append(cur_bal)
            
        bakiyeler = np.array(bakiyeler)
        zirveler = np.maximum.accumulate(bakiyeler)
        drawdowns = (bakiyeler - zirveler) / zirveler * 100.0
        max_drawdown = drawdowns.min()
        mean_holding = df_trades['Elde_Tutma_Gun'].mean()
    else:
        max_drawdown = 0.0
        mean_holding = 0.0
        
    print("\n" + "="*60)
    print(f"=== HİBRİT YAPAY ZEKA MASTER BEYİN BACKTEST RAPORU ({etiket}) ===")
    print("="*60)
    print(f"Simülasyon Aralığı: 1 Ocak 2016 -> {end_date.strftime('%d-%m-%Y')}")
    print(f"Başlangıç Sermayesi: $10,000.00 USD")
    print(f"Toplam Yapılan İşlem Sayısı: {total_trades}")
    print(f"   Karlı Tamamlanan (Win): {wins}")
    print(f"   Zararla Kapanan (Loss): {losses}")
    print(f"   Yapay Zeka Başarı Oranı (Win Rate): %{win_rate:.2f}")
    print(f"Ortalama Pozisyonda Kalma Süresi: {mean_holding:.1f} Gün")
    print(f"Maksimum Çekilme (Max Drawdown): %{max_drawdown:.2f}")
    print("-"*60)
    print(f"💵 HİBRİT YAPAY ZEKA PORTFÖY DEĞERİ: ${bakiye_son:,.2f} USD")
    print(f"📈 Yapay Zeka Portföy Net Getirisi: %{model_getiri_pct:+.2f} (${model_net_kar:+,.2f} USD)")
    print("-"*60)
    print(f"📊 BUY & HOLD (Simple Index) PORTFÖYÜ: ${buy_hold_final:,.2f} USD")
    print(f"📉 Buy & Hold Eşit Ağırlıklı Getiri: %{buy_hold_getiri_pct:+.2f} (${(buy_hold_final - baslangic_bakiye):+,.2f} USD)")
    print("="*60)
    
    print("\n[+] İşlem Yönlerinin Dağılımı:")
    if total_trades > 0 and 'Yon' in df_trades.columns:
        yon_counts = df_trades['Yon'].value_counts()
        for y_name, y_count in yon_counts.items():
            print(f"   {y_name:10} : {y_count} adet")
            
    print("\n[+] İşlem Kapanış Sebeplerinin Dağılımı:")
    if total_trades > 0:
        sebep_counts = df_trades['Sebep'].value_counts()
        for s_name, s_count in sebep_counts.items():
            print(f"   {s_name:30} : {s_count} adet")
    else:
        print("   Hiç işlem yapılmadı.")
    print("="*60)

metrikleri_yazdir(bakiye_strict, trades_strict, "ORİJİNAL SIKI MOD - %80")
metrikleri_yazdir(bakiye_calib, trades_calib, "CALİBRE EDİLMİŞ OPTİMİZE MOD - %52")

print("\n[BILGI] Ayrıntılı calibre edilmiş tarihsel işlem dökümü 'geriye_donuk_backtest_detayli.csv' olarak diske kaydedildi.")
print("==================================================")
