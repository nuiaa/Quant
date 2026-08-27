import os
import sys
import time
import json
import pickle
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

from beyin_mimarisi import DinamikHiyerarsikModel
from sicak_motor import veriyi_oku_ve_ozellikleri_hesapla

print("==================================================")
print("=== YENİ DİNAMİK AHTAPOT BEYİN GERİYE DÖNÜK BACKTEST ===")
print("==================================================")

Cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DONANIM] Çıkarım için kullanılan donanım: {Cihaz}")

# Sektör Haritası Yükle
sektor_haritasi = {}
if os.path.exists("piyasa_haritasi.json"):
    try:
        with open("piyasa_haritasi.json", "r", encoding="utf-8") as f:
            piyasa = json.load(f)
        sektorler = piyasa.get("SEKTORLER", {})
        for s_adi, s_v in sektorler.items():
            for end_adi, end_v in s_v.get("Endustriler", {}).items():
                for hisse in end_v.get("Hisseler", []):
                    sektor_haritasi[hisse] = s_adi
    except Exception as e:
        print(f"[UYARI] piyasa_haritasi.json okunamadı: {e}")

SEMBOL_LISTESI = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'V', 'JNJ', 'XOM', 'CAT', 'SPY']
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

# Model ve Scaler Cache
model_cache = {}
def modeli_yukle(model_adi, islem_yonu):
    anahtar = f"{model_adi}_{islem_yonu}"
    if anahtar in model_cache:
        return model_cache[anahtar]

    pth = f"{model_adi}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"
    sm = f"{model_adi}_scaler_makro.pkl"
    st = f"{model_adi}_scaler_teknik.pkl"

    if not (os.path.exists(pth) and os.path.exists(sm) and os.path.exists(st)):
        return None, None, None

    model = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)
    model.load_state_dict(torch.load(pth, map_location=Cihaz))
    model.eval()

    with open(sm, "rb") as f: scaler_makro = pickle.load(f)
    with open(st, "rb") as f: scaler_teknik = pickle.load(f)

    model_cache[anahtar] = (model, scaler_makro, scaler_teknik)
    return model, scaler_makro, scaler_teknik

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
    
    sektor_adi = sektor_haritasi.get(sembol, "BILINMEYEN")
    model_adi = sektor_adi.replace(" ", "_").upper()
    
    m_long, sm_long, st_long = modeli_yukle(model_adi, "LONG")
    m_short, sm_short, st_short = modeli_yukle(model_adi, "SHORT")
    
    if m_long is None or m_short is None:
        print(f"   [UYARI] {sembol} için {model_adi} modeli bulunamadı, tahmin yapılamıyor.")
        if sembol == 'SPY':
            predict_data['SPY'] = df.loc[df.index >= '2024-01-01'].copy()
        continue
        
    X_m_raw = df[makro_ozellikler].values
    X_t_raw = df[teknik_ozellikler].values
    
    X_m_sc_l = sm_long.transform(X_m_raw)
    X_t_sc_l = st_long.transform(X_t_raw)
    
    X_m_sc_s = sm_short.transform(X_m_raw)
    X_t_sc_s = st_short.transform(X_t_raw)
    
    df['Stop_Loss'] = df['Close'] - (df['ATR'] * 2.0)
    dinamik_hedef = df['Close'] + (df['ATR'] * 3.0)
    klasik_direnc = df['High'].rolling(window=20).max()
    df['Direnc_Hedefi'] = pd.concat([dinamik_hedef, klasik_direnc], axis=1).max(axis=1)
    
    ai_probs_long = np.zeros(len(df))
    ai_probs_short = np.zeros(len(df))
    N = len(df)
    
    # Sequence prediction
    GECMIS_MUM_SAYISI = 60
    
    from tqdm import tqdm
    for idx in tqdm(range(GECMIS_MUM_SAYISI, N), desc="Yapay Zeka Tahminleri"):
        # Long
        m_seq_l = torch.tensor(X_m_sc_l[idx-1:idx], dtype=torch.float32).to(Cihaz)
        t_seq_l = torch.tensor(X_t_sc_l[idx-GECMIS_MUM_SAYISI:idx][np.newaxis, ...], dtype=torch.float32).to(Cihaz)
        
        # Short
        m_seq_s = torch.tensor(X_m_sc_s[idx-1:idx], dtype=torch.float32).to(Cihaz)
        t_seq_s = torch.tensor(X_t_sc_s[idx-GECMIS_MUM_SAYISI:idx][np.newaxis, ...], dtype=torch.float32).to(Cihaz)
        
        with torch.no_grad():
            out_l = torch.sigmoid(m_long(m_seq_l, t_seq_l)).cpu().item()
            out_s = torch.sigmoid(m_short(m_seq_s, t_seq_s)).cpu().item()
            
        ai_probs_long[idx] = out_l
        ai_probs_short[idx] = out_s
            
    df['AI_Olasiligi_Long'] = ai_probs_long * 100.0
    df['AI_Olasiligi_Short'] = ai_probs_short * 100.0
    
    # 2024 filtering
    df_filtered = df.loc[df.index >= '2024-01-01'].copy()
    predict_data[sembol] = df_filtered
    sure_sn = time.time() - t_basla
    
    if sembol != 'SPY':
        probs_2024 = df_filtered['AI_Olasiligi_Long'].values
        p_min, p_max, p_mean = probs_2024.min(), probs_2024.max(), probs_2024.mean()
        print(f"   {sembol:6} | Çıkarım tamam. Model: {model_adi} | Süre: {sure_sn:.1f} sn | Ort_Long_Prob: %{p_mean:.1f}")

# KRONOLOJİK SİMÜLASYON FONKSİYONU
if not predict_data:
    print("[HATA] Tahmin yapılmış veri yok, simülasyon iptal.")
    sys.exit(1)
    
all_dates = sorted(list(set().union(*[predict_data[s].index for s in predict_data])))
baslangic_bakiye = 10000.0
risk_orani = 0.02

def simulasyon_calistir(al_esigi, sat_esigi):
    bakiye = baslangic_bakiye
    nakit = baslangic_bakiye
    aktif_pozisyonlar = []
    islem_gecmisi = []
    
    for gun in all_dates:
        # A. AKTİF POZİSYON GÜNCELLEME
        guncel_pozisyonlar = []
        for pos in aktif_pozisyonlar:
            sembol = pos['sembol']
            df_sembol = predict_data[sembol]
            
            if gun not in df_sembol.index:
                guncel_pozisyonlar.append(pos)
                continue
                
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
            
            # Yeni sınır: %10'un altına düşerse veya zıt yön %85'i geçerse kaç.
            AI_PANIK_ESIGI = 10.0
            esik_limit = 85.0
            if pos.get('yon', 'LONG') == 'LONG':
                pos['max_high'] = max(pos['max_high'], high)
                if low <= pos['stop_loss']:
                    cikis_fiyati = pos['stop_loss']
                    sebep = "🛑 Stop-Loss (Mekanik)"
                    cikis_yapildi = True
                elif high >= pos['target']:
                    cikis_fiyati = pos['target']
                    sebep = "✅ Take-Profit (Mekanik)"
                    cikis_yapildi = True
                elif ai_prob_long < AI_PANIK_ESIGI or ai_prob_short >= esik_limit:
                    cikis_fiyati = close
                    sebep = f"🧠 AI Erken Kaçış (Skor: %{ai_prob_long:.1f})"
                    cikis_yapildi = True
                elif pos['days_held'] > 15:
                    cikis_fiyati = close
                    sebep = "ZAMAN AŞIMI"
                    cikis_yapildi = True
            else:
                pos['min_low'] = min(pos['min_low'], low)
                if high >= pos['stop_loss']:
                    cikis_fiyati = pos['stop_loss']
                    sebep = "🛑 Stop-Loss SHORT (Mekanik)"
                    cikis_yapildi = True
                elif low <= pos['target']:
                    cikis_fiyati = pos['target']
                    sebep = "✅ Take-Profit SHORT (Mekanik)"
                    cikis_yapildi = True
                elif ai_prob_short < AI_PANIK_ESIGI or ai_prob_long >= esik_limit:
                    cikis_fiyati = close
                    sebep = f"🧠 AI Erken Kaçış (Skor: %{ai_prob_short:.1f})"
                    cikis_yapildi = True
                elif pos['days_held'] > 15:
                    cikis_fiyati = close
                    sebep = "ZAMAN AŞIMI"
                    cikis_yapildi = True
                                
            if cikis_yapildi:
                if pos.get('yon', 'LONG') == 'LONG':
                    tutar = pos['qty'] * cikis_fiyati
                    nakit += tutar
                    pl = tutar - (pos['qty'] * pos['entry_price'])
                else: 
                    pl = (pos['entry_price'] - cikis_fiyati) * pos['qty']
                    tutar = (pos['qty'] * pos['entry_price']) + pl
                    nakit += tutar
                
                pl_pct = pl / (pos['qty'] * pos['entry_price']) * 100.0
                islem_gecmisi.append({
                    'Sembol': sembol,
                    'Yon': pos.get('yon', 'LONG'),
                    'Giris_Tarihi': pos['entry_date'].strftime('%Y-%m-%d'),
                    'Cikis_Tarihi': gun.strftime('%Y-%m-%d'),
                    'Adet': pos['qty'],
                    'Giris_Fiyati': pos['entry_price'],
                    'Cikis_Fiyati': cikis_fiyati,
                    'Hedef_Fiyati': pos['target'],
                    'Stop_Fiyati': pos['stop_loss'],
                    'P&L_USD': pl,
                    'Sebep': sebep,
                    'Elde_Tutma_Gun': pos['days_held']
                })
            else:
                guncel_pozisyonlar.append(pos)
                
        aktif_pozisyonlar = guncel_pozisyonlar
        
        # B. KASA PORTFÖY
        portfoy_degeri = nakit
        for pos in aktif_pozisyonlar:
            sembol = pos['sembol']
            df_sembol = predict_data[sembol]
            c_price = float(df_sembol.loc[gun]['Close']) if gun in df_sembol.index else pos['entry_price']
            if pos.get('yon', 'LONG') == 'LONG':
                portfoy_degeri += pos['qty'] * c_price
            else:
                pl_now = (pos['entry_price'] - c_price) * pos['qty']
                portfoy_degeri += (pos['qty'] * pos['entry_price']) + pl_now
        bakiye = portfoy_degeri
        
        # C. ALIM SİNYALLERİ
        adaylar = []
        spy_df = predict_data.get('SPY', None)
        spy_trend_long_ok = True
        spy_trend_short_ok = True
        
        # MAKRO TREND FİLTRESİ (S&P 500 200 Günlük Hareketli Ortalama)
        if spy_df is not None and gun in spy_df.index:
            spy_ema_200_farki = float(spy_df.loc[gun].get('Fiyat_EMA200_Farki', 0.0))
            if spy_ema_200_farki < 0:
                spy_trend_long_ok = False # SPY 200 SMA altındaysa Long YASAK
            if spy_ema_200_farki > 0:
                spy_trend_short_ok = False # SPY 200 SMA üzerindeyse Short YASAK

        for sembol in predict_data:
            if sembol == 'SPY': continue # SPY'ı alıp satmıyoruz, sadece pusula
            df_sembol = predict_data[sembol]
            if gun not in df_sembol.index: continue
            if any(pos['sembol'] == sembol for pos in aktif_pozisyonlar): continue
                
            row = df_sembol.loc[gun]
            ai_prob_long = float(row['AI_Olasiligi_Long'])
            ai_prob_short = float(row['AI_Olasiligi_Short'])
            esik_limit_long = al_esigi
            esik_limit_short = 85.0

            if ai_prob_long >= esik_limit_long and ai_prob_short < 40.0 and spy_trend_long_ok:
                adaylar.append({'sembol': sembol, 'yon': 'LONG', 'row': row, 'ai_prob': ai_prob_long})
            elif ai_prob_short >= esik_limit_short and ai_prob_long < 40.0 and spy_trend_short_ok:
                adaylar.append({'sembol': sembol, 'yon': 'SHORT', 'row': row, 'ai_prob': ai_prob_short})
                
        adaylar = sorted(adaylar, key=lambda x: x['ai_prob'], reverse=True)
        for aday in adaylar:
            sembol = aday['sembol']
            yon = aday['yon']
            row = aday['row']
            close_price = float(row['Close'])
            atr = float(row['ATR'])
            
            if yon == 'LONG':
                stop_loss = close_price - (atr * 2.5)
                target = close_price + (atr * 2.5)
                risk_miktari = close_price - stop_loss
            else:
                stop_loss = close_price + (atr * 2.5)
                target = close_price - (atr * 2.5)
                risk_miktari = stop_loss - close_price
            
            if risk_miktari <= 0: continue
            
            # Kasa %30 Nakit Kalkanı (Dokunulmaz)
            minimum_nakit_siniri = baslangic_bakiye * 0.30
            kullanilabilir_nakit = max(0, nakit - minimum_nakit_siniri)
            
            risklenecek_nakit = bakiye * risk_orani
            adet = int(risklenecek_nakit / risk_miktari)
            if adet <= 0: adet = 1
                
            gerekli_para = adet * close_price
            
            # Sadece KULLANILABİLİR nakit üzerinden alım yap
            if gerekli_para > kullanilabilir_nakit:
                adet = int(kullanilabilir_nakit / close_price)
                gerekli_para = adet * close_price
                
            if adet > 0 and gerekli_para <= nakit:
                nakit -= gerekli_para
                aktif_pozisyonlar.append({
                    'sembol': sembol, 'yon': yon, 'entry_date': gun, 'entry_price': close_price,
                    'stop_loss': stop_loss, 'target': target, 'qty': adet, 'entry_atr': atr,
                    'max_high': close_price, 'min_low': close_price, 'days_held': 0
                })
                
    trades_df = pd.DataFrame(islem_gecmisi)
    return bakiye, trades_df

print(f"\n[SİMÜLASYON] Keskin Nişancı Modu (Alım %75.0 / Çıkış %50.0)...")
bakiye_calib, trades_calib = simulasyon_calistir(75.0, 50.0)

def metrikleri_yazdir(bakiye_son, df_trades, etiket):
    total_trades = len(df_trades) if not df_trades.empty else 0
    wins = len(df_trades[df_trades['P&L_USD'] >= 0]) if total_trades > 0 else 0
    losses = len(df_trades[df_trades['P&L_USD'] < 0]) if total_trades > 0 else 0
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    
    model_net_kar = bakiye_son - baslangic_bakiye
    model_getiri_pct = model_net_kar / baslangic_bakiye * 100.0
    
    print("\n" + "="*60)
    print(f"=== HİBRİT YAPAY ZEKA MASTER BEYİN BACKTEST RAPORU ({etiket}) ===")
    print(f"Toplam Yapılan İşlem Sayısı: {total_trades}")
    print(f"   Yapay Zeka Başarı Oranı (Win Rate): %{win_rate:.2f}")
    print("-"*60)
    print(f"💵 HİBRİT YAPAY ZEKA PORTFÖY DEĞERİ: ${bakiye_son:,.2f} USD")
    print(f"📈 Yapay Zeka Portföy Net Getirisi: %{model_getiri_pct:+.2f} (${model_net_kar:+,.2f} USD)")
    print("="*60)

metrikleri_yazdir(bakiye_calib, trades_calib, "Optimize Mod")
if not trades_calib.empty:
    print("\n--- ÇIKIŞ SEBEPLERİ ---")
    print(trades_calib['Sebep'].value_counts())
    
    # Tüm işlemleri CSV'ye dök (Analiz için)
    trades_calib.to_csv("sanal_islem_gecmisi.csv", index=False)
