import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

# Reconfigure stdout for utf-8 safe terminal handling
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sicak_motor import veriyi_oku_ve_ozellikleri_hesapla, tensore_donustur_hiyerarsik
from beyin_mimarisi import DinamikHiyerarsikModel

def numpy_auc(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    pos = np.sort(pos)
    neg = np.sort(neg)
    ranks_left = np.searchsorted(pos, neg, side='left')
    ranks_right = np.searchsorted(pos, neg, side='right')
    u = np.sum((len(pos) - ranks_left) + (len(pos) - ranks_right)) / 2.0
    auc = u / (len(pos) * len(neg))
    return auc

def hesapla_metrikler(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    y_pred_bin = (y_pred >= threshold).astype(float)
    
    tp = np.sum((y_pred_bin == 1.0) & (y_true == 1.0))
    fp = np.sum((y_pred_bin == 1.0) & (y_true == 0.0))
    tn = np.sum((y_pred_bin == 0.0) & (y_true == 0.0))
    fn = np.sum((y_pred_bin == 0.0) & (y_true == 1.0))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Acc': accuracy, 'Prec': precision, 'Recall': recall, 'F1': f1
    }

def model_test_et(sembol, islem_yonu, islenmis_veri, model_adi):
    print("-" * 75)
    print(f"📊 {sembol} - {islem_yonu} Ahtapot Beyni Başarı Oranları Analiz Ediliyor...")
    print("-" * 75)
    
    if not TORCH_AVAILABLE:
        print("[HATA] PyTorch kurulu olmadığı için analiz gerçekleştirilemiyor!")
        return
        
    model_dosyasi = f"{model_adi}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"
    if not os.path.exists(model_dosyasi):
        print(f"[UYARI] {model_dosyasi} bulunamadı! Geçiliyor...")
        return
        
    # tensore_donustur_hiyerarsik artık 9-tuple döndürüyor
    sonuc = tensore_donustur_hiyerarsik(islenmis_veri, islem_yonu=islem_yonu)
    if sonuc is None:
        print("[HATA] Veri yetersiz!")
        return
    X_makro, X_teknik, Y_tensor = sonuc[0], sonuc[1], sonuc[2]
    
    model = DinamikHiyerarsikModel(makro_girdi_sayisi=9, teknik_girdi_sayisi=15)
    model.load_state_dict(torch.load(model_dosyasi, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        preds = torch.sigmoid(model(X_makro, X_teknik)).cpu().numpy().flatten()
        
    y_true = Y_tensor.cpu().numpy().flatten()
    auc_score = numpy_auc(y_true, preds)
    
    print(f"[BİLGİ] Dengelenmiş Küme Sınıf Dağılımı -> 1: {np.sum(y_true==1):4} | 0: {np.sum(y_true==0):4}")
    print(f"[METRİK] ROC-AUC Skoru: %{auc_score*100.0:.2f}")
    
    esikler = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    
    print("\n📈 Eşik Değerlerine Göre Detaylı Performans Tablosu:")
    print("=" * 110)
    print(f"{'Eşik':6} | {'TP':6} | {'FP':6} | {'TN':6} | {'FN':6} | {'Accuracy':9} | {'Precision':9} | {'Recall':9} | {'F1-Score':9}")
    print("=" * 110)
    
    for thr in esikler:
        m = hesapla_metrikler(y_true, preds, threshold=thr)
        print(f"{thr:.2f}  | {m['TP']:6} | {m['FP']:6} | {m['TN']:6} | {m['FN']:6} | %{m['Acc']*100:.1f}    | %{m['Prec']*100:.1f}    | %{m['Recall']*100:.1f}    | %{m['F1']*100:.1f}")
        print("-" * 110)

def tekil_sembol_metrikleri_hesapla(sembol, islenmis_veri, islem_yonu, model_adi, threshold):
    scaler_makro_dosya = f"{model_adi}_scaler_makro.pkl"
    scaler_teknik_dosya = f"{model_adi}_scaler_teknik.pkl"
    
    if not os.path.exists(scaler_makro_dosya) or not os.path.exists(scaler_teknik_dosya):
        return None
        
    try:
        with open(scaler_makro_dosya, "rb") as f:
            scaler_makro = pickle.load(f)
        with open(scaler_teknik_dosya, "rb") as f:
            scaler_teknik = pickle.load(f)
    except:
        return None
        
    model_dosyasi = f"{model_adi}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"
    if not os.path.exists(model_dosyasi):
        return None
        
    if not TORCH_AVAILABLE:
        return None
        
    model = DinamikHiyerarsikModel(makro_girdi_sayisi=2, teknik_girdi_sayisi=15)
    model.load_state_dict(torch.load(model_dosyasi, map_location='cpu'))
    model.eval()
    
    makro_ozellikler = ['VIX', 'DXY']
    teknik_ozellikler = [
        'Fiyat_EMA20_Farki', 'Fiyat_EMA200_Farki', 'RSI', 
        'MACD', 'MACD_Histogram', 'ATR_Yuzde', 'BB_Pozisyon', 
        'Ust_Fitil_Gucu', 'Alt_Fitil_Gucu',       
        'Hacim_Patlamasi_Orani',                  
        'Fiyat_Degisimi_5G', 'RSI_Degisimi_5G',
        'Uyumsuzluk_Skoru', 'Likidite_Avi_Siddeti', 'BB_Sikisma_Orani'
    ]
    
    if islenmis_veri.empty or len(islenmis_veri) < 61:
        return None
        
    X_makro_ham = islenmis_veri[makro_ozellikler].values
    X_teknik_ham = islenmis_veri[teknik_ozellikler].values
    
    target_col = 'Hedef_Yonu_Long' if islem_yonu == "LONG" else 'Hedef_Yonu_Short'
    if target_col not in islenmis_veri.columns:
        return None
        
    Y_ham = islenmis_veri[target_col].values
    
    X_makro_olcekli = scaler_makro.transform(X_makro_ham)
    X_teknik_olcekli = scaler_teknik.transform(X_teknik_ham)
    
    GECMIS_MUM_SAYISI = 60
    X_makro_list, X_teknik_list, Y_list = [], [], []
    for i in range(len(islenmis_veri) - GECMIS_MUM_SAYISI):
        pencere_teknik = X_teknik_olcekli[i : (i + GECMIS_MUM_SAYISI)]
        pencere_makro = X_makro_olcekli[i + GECMIS_MUM_SAYISI - 1]
        hedef = Y_ham[i + GECMIS_MUM_SAYISI]
        X_teknik_list.append(pencere_teknik)
        X_makro_list.append(pencere_makro)
        Y_list.append(hedef)
        
    n_seq = len(Y_list)
    if n_seq == 0:
        return None
        
    # Validasyon kısmını al (%15)
    n_train = int(n_seq * 0.85)
    X_makro_val = np.array(X_makro_list[n_train:])
    X_teknik_val = np.array(X_teknik_list[n_train:])
    Y_val = np.array(Y_list[n_train:])
    
    X_makro_tensor = torch.tensor(X_makro_val, dtype=torch.float32)
    X_teknik_tensor = torch.tensor(X_teknik_val, dtype=torch.float32)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(X_makro_tensor, X_teknik_tensor)).cpu().numpy().flatten()
        
    m = hesapla_metrikler(Y_val, preds, threshold=threshold / 100.0)
    
    return {
        'TP': m['TP'], 'FP': m['FP'], 'TN': m['TN'], 'FN': m['FN'],
        'Acc': m['Acc'], 'Prec': m['Prec'], 'Recall': m['Recall'], 'F1': m['F1'],
        'SampleCount': len(Y_val)
    }

if __name__ == "__main__":
    print("=" * 90)
    print("🔥 QUANT HIYERARSIK (OCTOPUS) YAPAY ZEKA BAŞARI ORANLARI DOĞRULAMA MOTORU 🔥")
    print("=" * 90)
    
    rapor_satirlari = []
    
    # 1. Tekil Varlıklar (XAUUSD ve XAGUSD)
    SEMBOL = "XAUUSD"
    if os.path.exists("yapay_zeka_veritabani.sqlite"):
        try:
            islenmis_df = veriyi_oku_ve_ozellikleri_hesapla(SEMBOL)
            if islenmis_df is not None and len(islenmis_df) > 100:
                print(f"\n[ANALİZ] {SEMBOL} Başarı Oranı Test Ediliyor...")
                m_long = tekil_sembol_metrikleri_hesapla(SEMBOL, islenmis_df, "LONG", SEMBOL, 65.0)
                m_short = tekil_sembol_metrikleri_hesapla(SEMBOL, islenmis_df, "SHORT", SEMBOL, 65.0)
                
                if m_long:
                    rapor_satirlari.append({
                        'Sembol': SEMBOL, 'Yön': 'LONG', 'Eşik': 65.0,
                        'TP': m_long['TP'], 'FP': m_long['FP'], 'TN': m_long['TN'], 'FN': m_long['FN'],
                        'Acc': m_long['Acc'], 'Prec': m_long['Prec'], 'Recall': m_long['Recall'], 'F1': m_long['F1']
                    })
                if m_short:
                    rapor_satirlari.append({
                        'Sembol': SEMBOL, 'Yön': 'SHORT', 'Eşik': 65.0,
                        'TP': m_short['TP'], 'FP': m_short['FP'], 'TN': m_short['TN'], 'FN': m_short['FN'],
                        'Acc': m_short['Acc'], 'Prec': m_short['Prec'], 'Recall': m_short['Recall'], 'F1': m_short['F1']
                    })
        except Exception as e:
            print(f"[UYARI] {SEMBOL} yüklenemedi veya test edilemedi: {e}")

    SEMBOL2 = "XAGUSD"
    if os.path.exists("yapay_zeka_veritabani.sqlite"):
        try:
            islenmis_df = veriyi_oku_ve_ozellikleri_hesapla(SEMBOL2)
            if islenmis_df is not None and len(islenmis_df) > 100:
                print(f"\n[ANALİZ] {SEMBOL2} (Gümüş) Başarı Oranı Test Ediliyor...")
                m_long = tekil_sembol_metrikleri_hesapla(SEMBOL2, islenmis_df, "LONG", SEMBOL2, 58.0)
                m_short = tekil_sembol_metrikleri_hesapla(SEMBOL2, islenmis_df, "SHORT", SEMBOL2, 58.0)
                
                if m_long:
                    rapor_satirlari.append({
                        'Sembol': SEMBOL2, 'Yön': 'LONG', 'Eşik': 58.0,
                        'TP': m_long['TP'], 'FP': m_long['FP'], 'TN': m_long['TN'], 'FN': m_long['FN'],
                        'Acc': m_long['Acc'], 'Prec': m_long['Prec'], 'Recall': m_long['Recall'], 'F1': m_long['F1']
                    })
                if m_short:
                    rapor_satirlari.append({
                        'Sembol': SEMBOL2, 'Yön': 'SHORT', 'Eşik': 58.0,
                        'TP': m_short['TP'], 'FP': m_short['FP'], 'TN': m_short['TN'], 'FN': m_short['FN'],
                        'Acc': m_short['Acc'], 'Prec': m_short['Prec'], 'Recall': m_short['Recall'], 'F1': m_short['F1']
                    })
        except Exception as e:
            print(f"[UYARI] {SEMBOL2} yüklenemedi veya test edilemedi: {e}")
            
    # 2. S&P 500 Hisseleri Teker Teker Analiz Ediliyor
    SEMBOL_LISTESI = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V']
    
    print("\n[ANALİZ] S&P 500 Varlıkları Teker Teker Analiz Ediliyor (Eşik: %58.0)...")
    for sembol in SEMBOL_LISTESI:
        try:
            df = veriyi_oku_ve_ozellikleri_hesapla(sembol)
            if df is not None and len(df) > 100:
                m_long = tekil_sembol_metrikleri_hesapla(sembol, df, "LONG", "sp500_genel", 58.0)
                m_short = tekil_sembol_metrikleri_hesapla(sembol, df, "SHORT", "sp500_genel", 58.0)
                
                if m_long:
                    rapor_satirlari.append({
                        'Sembol': sembol, 'Yön': 'LONG', 'Eşik': 58.0,
                        'TP': m_long['TP'], 'FP': m_long['FP'], 'TN': m_long['TN'], 'FN': m_long['FN'],
                        'Acc': m_long['Acc'], 'Prec': m_long['Prec'], 'Recall': m_long['Recall'], 'F1': m_long['F1']
                    })
                if m_short:
                    rapor_satirlari.append({
                        'Sembol': sembol, 'Yön': 'SHORT', 'Eşik': 58.0,
                        'TP': m_short['TP'], 'FP': m_short['FP'], 'TN': m_short['TN'], 'FN': m_short['FN'],
                        'Acc': m_short['Acc'], 'Prec': m_short['Prec'], 'Recall': m_short['Recall'], 'F1': m_short['F1']
                    })
        except Exception as e:
            print(f"[UYARI] {sembol} analiz edilemedi: {e}")
            
    # Sonuç Raporunu Yazdır
    print("\n" + "=" * 115)
    print("📋 AHTAPOT HIYERARSIK YAPAY ZEKA VARLIK BAZLI NET BAŞARI ORANLARI TABLOSU (VALİDASYON VERİSİ)")
    print("=" * 115)
    print(f"{'Varlık':8} | {'Yön':5} | {'Eşik':6} | {'TP':4} | {'FP':4} | {'TN':5} | {'FN':4} | {'Accuracy':9} | {'Win Rate (Prec)':16} | {'Recall':7} | {'F1-Score':8}")
    print("=" * 115)
    
    for r in rapor_satirlari:
        print(f"{r['Sembol']:8} | {r['Yön']:5} | %{r['Eşik']:.1f} | {r['TP']:4} | {r['FP']:4} | {r['TN']:5} | {r['FN']:4} | %{r['Acc']*100:6.1f} | %{r['Prec']*100:13.1f} | %{r['Recall']*100:5.1f} | %{r['F1']*100:6.1f}")
        print("-" * 115)
    
    print("\n[BİLGİ] Win Rate (Precision), modelin sinyal ürettiği (TP+FP) işlemler arasında gerçekte kazananların (TP) oranını gösterir.")
    print("Win Rate ne kadar yüksekse, yanlış sinyaller ve komisyon kayıpları o derece azalmaktadır.")
