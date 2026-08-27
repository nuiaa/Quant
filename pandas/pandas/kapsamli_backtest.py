"""
KAPSAMLI BACKTEST & SANAL GERÇEK ZAMANLI TEST MOTORU
=====================================================
Veritabanındaki tüm 500+ varlık için:
  1. Model tahmini (sp500_genel modeli veya özel model)
  2. Validation seti üzerinde Win Rate, Accuracy, F1 hesaplama
  3. Sonuçları CSV'ye kaydetme
  4. Sektör bazlı özet rapor
"""
import os
import sys
import pickle
import sqlite3
import numpy as np
import pandas as pd
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sicak_motor import veriyi_oku_ve_ozellikleri_hesapla
from beyin_mimarisi import DinamikHiyerarsikModel

VERITABANI = "yapay_zeka_veritabani.sqlite"
THRESHOLD  = 0.50   # Eşik düşürüldü: pos_weight+sigmoid ile model artık cesur sinyal veriyor
GECMIS_MUM = 60

MAKRO_OZL = ['VIX', 'DXY']
TEKNIK_OZL = [
    'Fiyat_EMA20_Farki', 'Fiyat_EMA200_Farki', 'RSI',
    'MACD', 'MACD_Histogram', 'ATR_Yuzde', 'BB_Pozisyon',
    'Ust_Fitil_Gucu', 'Alt_Fitil_Gucu',
    'Hacim_Patlamasi_Orani',
    'Fiyat_Degisimi_5G', 'RSI_Degisimi_5G',
    'Uyumsuzluk_Skoru', 'Likidite_Avi_Siddeti', 'BB_Sikisma_Orani'
]

# ======================================================
# 1. TÜM 1d VARLIKLARINI VERİTABANINDAN OKU
# ======================================================
def db_sembol_listesi_cek():
    """Veritabanındaki tüm benzersiz 1d hisse sembollerini döndürür."""
    conn = sqlite3.connect(VERITABANI)
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'cache_%_1d'"
    tablolar = [r[0] for r in conn.execute(sql).fetchall()]
    conn.close()

    # Hariç tutulacaklar: metadata, makro endeksler, ETF'ler (opsiyonel)
    harici = {
        'cache_metadata', 'cache__VIX_1d', 'cache_DX_Y_NYB_1d', 'cache_DX_Y.NYB_1d',
        'cache_SPY_1d', 'cache_XLB_1d', 'cache_XLC_1d', 'cache_XLE_1d', 'cache_XLF_1d',
        'cache_XLI_1d', 'cache_XLK_1d', 'cache_XLP_1d', 'cache_XLRE_1d', 'cache_XLU_1d',
        'cache_XLV_1d', 'cache_XLY_1d', 'cache_GC_F_1d', 'cache_SI_F_1d', 'cache_PL_F_1d',
        'cache_PA_F_1d', 'cache_HG_F_1d', 'cache_NG_F_1d', 'cache_DBC_1d', 'cache_PSKY_1d'
    }

    semboller = []
    for t in tablolar:
        if t in harici:
            continue
        # cache_{SEMBOL}_1d  →  SEMBOL
        sembol = t.replace("cache_", "").replace("_1d", "")
        semboller.append(sembol)

    return sorted(set(semboller))


# ======================================================
# 2. MODEL YÜKLEYİCİ (CACHE'Lİ)
# ======================================================
model_cache = {}
scaler_cache = {}

def modeli_yukle(model_adi, islem_yonu):
    """Modeli ve scaler'ları bellekte cache'leyerek yükler."""
    anahtar = f"{model_adi}_{islem_yonu}"
    if anahtar in model_cache:
        return model_cache[anahtar], scaler_cache[anahtar]

    pth = f"{model_adi}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"
    sm  = f"{model_adi}_scaler_makro.pkl"
    st  = f"{model_adi}_scaler_teknik.pkl"

    if not (os.path.exists(pth) and os.path.exists(sm) and os.path.exists(st)):
        return None, None

    if not TORCH_AVAILABLE:
        return None, None

    model = DinamikHiyerarsikModel(makro_girdi_sayisi=9, teknik_girdi_sayisi=15)
    model.load_state_dict(torch.load(pth, map_location='cpu'))
    model.eval()

    with open(sm, "rb") as f: scaler_makro = pickle.load(f)
    with open(st, "rb") as f: scaler_teknik = pickle.load(f)

    model_cache[anahtar]  = model
    scaler_cache[anahtar] = (scaler_makro, scaler_teknik)
    return model, (scaler_makro, scaler_teknik)


# ======================================================
# 3. TEKİL VARLIK DEĞERLENDİRME
# ======================================================
def varlik_degerlendir(sembol, islem_yonu, df, model_adi, thr=THRESHOLD):
    """
    Varlık için hem train hem de validation metriklerini hesaplar.
    Geri döner: dict veya None
    """
    model, scalers = modeli_yukle(model_adi, islem_yonu)
    if model is None or df is None or df.empty or len(df) < GECMIS_MUM + 10:
        return None

    scaler_makro, scaler_teknik = scalers

    # Gerekli sütunlar var mı?
    hedef_col = 'Hedef_Yonu_Long' if islem_yonu == 'LONG' else 'Hedef_Yonu_Short'
    eksik = [c for c in MAKRO_OZL + TEKNIK_OZL + [hedef_col] if c not in df.columns]
    if eksik:
        return None

    X_m_raw = df[MAKRO_OZL].values
    X_t_raw = df[TEKNIK_OZL].values
    Y_raw   = df[hedef_col].values

    X_m_sc = scaler_makro.transform(X_m_raw)
    X_t_sc = scaler_teknik.transform(X_t_raw)

    # Kayan pencere
    makro_seq, teknik_seq, y_seq = [], [], []
    for i in range(len(df) - GECMIS_MUM):
        teknik_seq.append(X_t_sc[i : i + GECMIS_MUM])
        makro_seq.append(X_m_sc[i + GECMIS_MUM - 1])
        y_seq.append(Y_raw[i + GECMIS_MUM])

    if len(y_seq) < 20:
        return None

    n_train = int(len(y_seq) * 0.85)
    n_val   = len(y_seq) - n_train

    # --- TRAIN metrikler ---
    def metrikleri_hesapla(m_arr, t_arr, y_arr, label):
        if len(y_arr) == 0:
            return {}
        X_m_t = torch.tensor(np.array(m_arr), dtype=torch.float32)
        X_t_t = torch.tensor(np.array(t_arr), dtype=torch.float32)
        with torch.no_grad():
            preds = torch.sigmoid(model(X_m_t, X_t_t)).numpy().flatten()
        y = np.array(y_arr)
        p_bin = (preds >= thr).astype(float)
        tp = int(np.sum((p_bin==1) & (y==1)))
        fp = int(np.sum((p_bin==1) & (y==0)))
        tn = int(np.sum((p_bin==0) & (y==0)))
        fn = int(np.sum((p_bin==0) & (y==1)))
        acc  = (tp+tn)/len(y) if len(y)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        # AUC (basit)
        pos_preds = preds[y==1]
        neg_preds = preds[y==0]
        if len(pos_preds)>0 and len(neg_preds)>0:
            auc = np.mean(pos_preds > neg_preds.mean())
        else:
            auc = 0.5
        return {
            f'{label}_N': len(y), f'{label}_TP': tp, f'{label}_FP': fp,
            f'{label}_TN': tn, f'{label}_FN': fn,
            f'{label}_Acc': round(acc*100,1),
            f'{label}_WinRate': round(prec*100,1),
            f'{label}_Recall': round(rec*100,1),
            f'{label}_F1': round(f1*100,1),
            f'{label}_AUC': round(auc*100,1),
        }

    train_m = metrikleri_hesapla(makro_seq[:n_train], teknik_seq[:n_train], y_seq[:n_train], 'Train')
    val_m   = metrikleri_hesapla(makro_seq[n_train:], teknik_seq[n_train:], y_seq[n_train:], 'Val')

    sonuc = {
        'Sembol': sembol, 'Yon': islem_yonu, 'Model': model_adi,
        'Esik': round(thr*100, 1),
        'Train_N': train_m.get('Train_N', 0),
        'Val_N': val_m.get('Val_N', 0),
    }
    sonuc.update(train_m)
    sonuc.update(val_m)
    return sonuc


# ======================================================
# 4. SEKTÖR HARİTASI
# ======================================================
import json
sektor_haritasi = {}  # sembol -> sektor_adi

if os.path.exists("piyasa_haritasi.json"):
    try:
        with open("piyasa_haritasi.json", "r", encoding="utf-8") as f:
            piyasa = json.load(f)
        sektorler = piyasa.get("SEKTORLER", {})
        for sektor_adi, sektor_v in sektorler.items():
            for endustri_adi, endustri_v in sektor_v.get("Endustriler", {}).items():
                for hisse in endustri_v.get("Hisseler", []):
                    sektor_haritasi[hisse] = sektor_adi
        print(f"[SEKTOR] {len(sektor_haritasi)} varl\u0131\u011f\u0131n sekt\u00f6r DNA's\u0131 y\u00fcklendi.")
    except Exception as e:
        print(f"[UYARI] piyasa_haritasi.json okunamad\u0131: {e}")


# ======================================================
# 5. ANA DÖNGÜ
# ======================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 90)
    print("🔥 KAPSAMLI 500+ VARLIK BACKTEST & SANAL GERÇEK ZAMANLI TEST MOTORU 🔥")
    print("=" * 90)

    semboller = db_sembol_listesi_cek()
    print(f"[BİLGİ] Veritabanında {len(semboller)} benzersiz varlık tespit edildi.")
    print(f"[BİLGİ] Kullanılan karar eşiği: %{THRESHOLD*100:.0f}")
    print(f"[BİLGİ] Modeller: sp500_genel (hisseler) | XAUUSD / XAGUSD (emtialar)")
    print("-" * 90)

    sonuclar = []
    hata_sayisi = 0
    islendi = 0

    for i, sembol in enumerate(semboller, 1):
        # Model seçimi
        if sembol in ('XAUUSD', 'XAGUSD'):
            model_adi = sembol
        else:
            model_adi = 'sp500_genel'

        try:
            df = veriyi_oku_ve_ozellikleri_hesapla(sembol)
        except Exception as e:
            hata_sayisi += 1
            continue

        if df is None or df.empty or len(df) < 80:
            hata_sayisi += 1
            continue

        for yon in ['LONG', 'SHORT']:
            r = varlik_degerlendir(sembol, yon, df, model_adi, thr=THRESHOLD)
            if r:
                r['Sektor'] = sektor_haritasi.get(sembol, 'Bilinmiyor')
                sonuclar.append(r)

        islendi += 1
        if i % 50 == 0:
            gecen = time.time() - t0
            print(f"[{i:3d}/{len(semboller)}] {islendi} varlık işlendi | {hata_sayisi} hata | {gecen:.0f}s")

    print(f"\n[TAMAMLANDI] {islendi} varlık başarıyla test edildi | {hata_sayisi} atlandı")
    print(f"[SÜRE] Toplam: {(time.time()-t0):.1f} saniye")

    if not sonuclar:
        print("[HATA] Hiç sonuç üretilemedi!")
        sys.exit(1)

    # ======================================================
    # 6. SONUÇ TABLOSU VE CSV
    # ======================================================
    df_rapor = pd.DataFrame(sonuclar)
    df_rapor.to_csv("backtest_sonuclari_tam.csv", index=False, encoding='utf-8-sig')
    print(f"\n[KAYIT] Tüm sonuçlar 'backtest_sonuclari_tam.csv' dosyasına kaydedildi.")

    # --- VALİDASYON bazlı özet ---
    val_cols = [c for c in df_rapor.columns if c.startswith('Val_')]
    if 'Val_WinRate' in df_rapor.columns and 'Val_N' in df_rapor.columns:
        df_val = df_rapor[df_rapor['Val_N'] >= 5].copy()

        print("\n" + "=" * 130)
        print("📊 VALİDASYON SETİ BAŞARI TABLSU (Gerçek Out-of-Sample Performans)")
        print("=" * 130)
        print(f"{'Sembol':10} {'Yön':6} {'Sektör':30} {'Val_N':6} {'Win%':7} {'Acc%':7} {'Recall%':8} {'F1%':6} {'Train_N':7} {'Train_Win%':10}")
        print("-" * 130)

        for _, row in df_val.sort_values('Val_WinRate', ascending=False).head(60).iterrows():
            print(f"{row['Sembol']:10} {row['Yon']:6} {str(row.get('Sektor','?'))[:30]:30} "
                  f"{int(row['Val_N']):6} "
                  f"%{row['Val_WinRate']:5.1f} "
                  f"%{row['Val_Acc']:5.1f} "
                  f"%{row.get('Val_Recall',0):5.1f} "
                  f"%{row.get('Val_F1',0):4.1f} "
                  f"{int(row['Train_N']):7} "
                  f"%{row.get('Train_WinRate',0):6.1f}")

        # --- SEKTÖR BAZLI ÖZET ---
        print("\n" + "=" * 80)
        print("🏭 SEKTÖR BAZLI ORTALAMA WIN RATE (Validasyon Seti)")
        print("=" * 80)
        sektor_ozet = df_val.groupby(['Sektor', 'Yon']).agg(
            Varlik_Sayisi=('Sembol', 'count'),
            Ort_WinRate=('Val_WinRate', 'mean'),
            Ort_Accuracy=('Val_Acc', 'mean'),
            Ort_F1=('Val_F1', 'mean'),
            Ort_Recall=('Val_Recall', 'mean'),
        ).reset_index().sort_values('Ort_WinRate', ascending=False)

        print(f"{'Sektör':35} {'Yön':6} {'Varlık':7} {'WinRate%':9} {'Accuracy%':10} {'F1%':6} {'Recall%':8}")
        print("-" * 80)
        for _, r in sektor_ozet.iterrows():
            print(f"{str(r['Sektor'])[:35]:35} {r['Yon']:6} {int(r['Varlik_Sayisi']):7} "
                  f"%{r['Ort_WinRate']:6.1f}  "
                  f"%{r['Ort_Accuracy']:7.1f}  "
                  f"%{r['Ort_F1']:4.1f}  "
                  f"%{r['Ort_Recall']:5.1f}")

        # --- ÖZET İSTATİSTİKLER ---
        print("\n" + "=" * 60)
        print("📈 GENEL ÖZET İSTATİSTİKLER")
        print("=" * 60)

        for yon in ['LONG', 'SHORT']:
            sub = df_val[df_val['Yon'] == yon]
            if sub.empty: continue
            print(f"\n  [{yon}] ({len(sub)} varlık test edildi)")
            print(f"  Ortalama Win Rate  : %{sub['Val_WinRate'].mean():.1f}")
            print(f"  Medyan Win Rate    : %{sub['Val_WinRate'].median():.1f}")
            print(f"  %60+ Win Rate olan : {(sub['Val_WinRate'] >= 60).sum()} varlık ({(sub['Val_WinRate'] >= 60).mean()*100:.0f}%)")
            print(f"  %55+ Win Rate olan : {(sub['Val_WinRate'] >= 55).sum()} varlık ({(sub['Val_WinRate'] >= 55).mean()*100:.0f}%)")
            print(f"  %50+ Win Rate olan : {(sub['Val_WinRate'] >= 50).sum()} varlık ({(sub['Val_WinRate'] >= 50).mean()*100:.0f}%)")
            print(f"  Ortalama Accuracy  : %{sub['Val_Acc'].mean():.1f}")
            print(f"  Ortalama F1        : %{sub['Val_F1'].mean():.1f}")

        # En iyi 15 LONG sinyali
        top_long  = df_val[df_val['Yon']=='LONG'].nlargest(15, 'Val_WinRate')
        top_short = df_val[df_val['Yon']=='SHORT'].nlargest(15, 'Val_WinRate')

        print("\n🏆 EN GÜÇLÜ 15 LONG VARLĞI (Validasyon Win Rate'e göre):")
        for _, r in top_long.iterrows():
            print(f"  {r['Sembol']:8} | Win: %{r['Val_WinRate']:5.1f} | Acc: %{r['Val_Acc']:5.1f} | F1: %{r['Val_F1']:5.1f} | N={int(r['Val_N'])} | {r.get('Sektor','?')}")

        print("\n🏆 EN GÜÇLÜ 15 SHORT VARLĞI (Validasyon Win Rate'e göre):")
        for _, r in top_short.iterrows():
            print(f"  {r['Sembol']:8} | Win: %{r['Val_WinRate']:5.1f} | Acc: %{r['Val_Acc']:5.1f} | F1: %{r['Val_F1']:5.1f} | N={int(r['Val_N'])} | {r.get('Sektor','?')}")

        sektor_ozet.to_csv("sektor_bazli_backtest_ozet.csv", index=False, encoding='utf-8-sig')
        print("\n[KAYIT] Sektör özeti 'sektor_bazli_backtest_ozet.csv' dosyasına kaydedildi.")

    print("\n=== KAPSAMLI BACKTEST TAMAMLANDI ===")
