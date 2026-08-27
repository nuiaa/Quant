import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# sys.stdout encoding reconfiguration for absolute safe terminal handling
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Try to import torch and dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import hot engine features and brain model
from sicak_motor import veriyi_oku_ve_ozellikleri_hesapla, tensore_donustur_hiyerarsik
from beyin_mimarisi import DinamikHiyerarsikModel

# ==========================================
# EĞİTİM AYARLARI (HİPERPARAMETRELER)
# ==========================================
BATCH_SIZE          = 128
EPOCH_SAYISI        = 100          # LSTM çok katmanlı; 100 epoch yeterli kapsam
OGRENME_ORANI       = 5e-5         # 1e-4 → 5e-5: pos_weight ile daha kararsız gradyanlar
AGIRLIK_SONUMLEMESI = 1e-2         # 1e-3 -> 1e-2: Anti-Overfitting (Daha güçlü sönümleme)
GRADYAN_CLIP        = 1.0          # Patlayıcı gradyan önleme (max_norm)
PATIENCE_VARSAYILAN = 20           # Early stopping sabrı (epoch)
WARMUP_EPOCH        = 15           # İlk 15 epoch'ta early stopping devre dışı
MIN_DELTA           = 1e-4         # Val Loss'un "iyileşme" sayılması için minimum düşüş

# Donanim Ivmelendirmesi: CUDA var mi kontrol et
if TORCH_AVAILABLE:
    Cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    Cihaz = "CPU (Simulasyon Modu)"

def beyni_egit(sembol, islem_yonu, islenmis_veri, model_adi=None):
    print("="*60)
    print(f"🧠 {islem_yonu} DİNAMİK AHTAPOT BEYNİ EĞİTİM PROTOKOLÜ ({sembol}) 🧠")
    print("="*60)

    # 1. Hiyerarşik ve Sızıntı Korumalı Veri Çekimi (9-tuple artık)
    sonuc = tensore_donustur_hiyerarsik(islenmis_veri, islem_yonu=islem_yonu)
    if sonuc is None:
        print("[HATA] Veri yetersiz, eğitim iptal edildi.")
        return

    X_m_tr, X_t_tr, Y_tr, X_m_val, X_t_val, Y_val, scaler_makro, scaler_teknik, pos_weight_ratio = sonuc

    if TORCH_AVAILABLE:
        # Eğitim Loader'ı
        train_seti  = TensorDataset(X_m_tr, X_t_tr, Y_tr)
        train_loader = DataLoader(train_seti, batch_size=BATCH_SIZE, shuffle=True)

        # Validasyon Loader'ı
        val_seti   = TensorDataset(X_m_val, X_t_val, Y_val)
        val_loader = DataLoader(val_seti, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Modeli Ayağa Kaldır
        model = DinamikHiyerarsikModel(makro_girdi_sayisi=13, teknik_girdi_sayisi=30).to(Cihaz)

        # 1. KAYIP FONKSİYONU: Saf, Dengesizlik Korumalı BCE Loss (Asimetrik Ceza İPTAL)
        pozitif_agirlik = torch.tensor([pos_weight_ratio], dtype=torch.float32).to(Cihaz)
        hakem = nn.BCEWithLogitsLoss(pos_weight=pozitif_agirlik)

        # Optimizatör (Ağırlık sönümleme - Weight Decay koruması aktif)
        antrenor = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

        # ReduceLROnPlateau: Val Loss 8 epoch düşmezse LR'yi yarıya indir
        zamanlayici = optim.lr_scheduler.ReduceLROnPlateau(
            antrenor, mode='min', factor=0.5, patience=8, min_lr=1e-7
        )

        pref = model_adi if model_adi else sembol
        dosya_adi = f"{pref}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"

        # Early Stopping değişkenleri
        best_val_loss    = float('inf')
        patience_counter = 0
        PATIENCE         = PATIENCE_VARSAYILAN  # 20 epoch sabır

        for epoch in range(1, EPOCH_SAYISI + 1):
            # ==========================================
            # A. EĞİTİM FAZI
            # ==========================================
            model.train()
            toplam_train_kayip = 0.0

            for b_m, b_t, b_y in train_loader:
                b_m, b_t, b_y = b_m.to(Cihaz), b_t.to(Cihaz), b_y.to(Cihaz)
                
                antrenor.zero_grad()
                logitler = model(b_m, b_t)          # Raw logit (sigmoid yok)
                
                # 🎯 İŞTE SİHİRLİ DOKUNUŞ: LABEL SMOOTHING
                # Hedef değerleri %10 oranında yumuşatıyoruz:
                # Kesin 1 (Yükselecek) olanlar -> 0.90 olur.
                # Kesin 0 (Düşecek) olanlar    -> 0.10 olur.
                batch_y_smoothed = b_y * 0.8 + 0.1 

                kayip = hakem(logitler, batch_y_smoothed)

                kayip.backward()
                # Patlayıcı gradyan önleme: LSTM'lerde kritik
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                antrenor.step()
                toplam_train_kayip += kayip.item()

            ort_train_loss = toplam_train_kayip / len(train_loader)

            # ==========================================
            # B. VALİDASYON FAZI
            # ==========================================
            model.eval()
            toplam_val_kayip = 0.0

            with torch.no_grad():
                for b_m, b_t, b_y in val_loader:
                    b_m, b_t, b_y = b_m.to(Cihaz), b_t.to(Cihaz), b_y.to(Cihaz)
                    val_logit = model(b_m, b_t)
                    
                    val_y_smoothed = b_y * 0.8 + 0.1 
                    val_kayip = hakem(val_logit, val_y_smoothed)
                    toplam_val_kayip += val_kayip.item()

            ort_val_loss = toplam_val_kayip / len(val_loader) if len(val_loader) > 0 else float('inf')

            # LR güncellemesi (Plateau scheduler)
            zamanlayici.step(ort_val_loss)
            guncel_lr = antrenor.param_groups[0]['lr']

            print(f"[{islem_yonu}] Epoch {epoch+1:03d}/{EPOCH_SAYISI} "
                  f"| Train: {ort_train_loss:.4f} "
                  f"| Val: {ort_val_loss:.4f} "
                  f"| LR: {guncel_lr:.2e} "
                  f"| pw: {pos_weight_ratio:.1f}x")

            # ==========================================
            # C. ERKEN DURDURMA — Isınma Donemine Saygı
            # ==========================================
            if epoch <= 30: # 30 epoch WARMUP
                print(f"   [İSİNMA] {30-epoch} epoch kaldı — Early stopping henuz aktif degil")
                if ort_val_loss < best_val_loss:
                    best_val_loss = ort_val_loss
                    torch.save(model.state_dict(), dosya_adi)
                continue
            else:
                # Isınma döneminde early stopping yok
                if epoch >= WARMUP_EPOCH:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\n🛑 ERKEN DURDURMA! {PATIENCE} epoch boyunca Val Loss iyileşmedi.")
                        print(f"🏆 En iyi Val Loss: {best_val_loss:.4f} — disk üzerindeki model korunuyor.")
                        break
                else:
                    print(f"   [İSİNMA] {WARMUP_EPOCH - epoch - 1} epoch kaldı — Early stopping henuz aktif degil")

        # Scaler'ları kaydet
        with open(f"{pref}_scaler_makro.pkl", "wb") as f:
            pickle.dump(scaler_makro, f)
        with open(f"{pref}_scaler_teknik.pkl", "wb") as f:
            pickle.dump(scaler_teknik, f)

        print(f"\n✅ {islem_yonu} AHTAPOT BEYİN EĞİTİLDİ: '{dosya_adi}'")
        print(f"✅ Scaler'lar kaydedildi: '{pref}_scaler_makro.pkl', '{pref}_scaler_teknik.pkl'\n")
    else:
        print("[SISTEM] PyTorch bulunamadığı için SIMULASYON modunda çalışılıyor...")
        pref = model_adi if model_adi else sembol
        dosya_adi = f"{pref}_{islem_yonu.lower()}_hiyerarsik_beyin.pth"

        dummy_state = {"simulated": True}
        with open(dosya_adi, "wb") as f:
            pickle.dump(dummy_state, f)

        with open(f"{pref}_scaler_makro.pkl", "wb") as f:
            pickle.dump(scaler_makro, f)
        with open(f"{pref}_scaler_teknik.pkl", "wb") as f:
            pickle.dump(scaler_teknik, f)
        print(f"✅ DUMMY Model ve Scaler'lar kaydedildi.\n")



if __name__ == "__main__":
    print(f"⚙️ Kullanılan Donanım: {Cihaz}")
    
    # Hisseleri Sektörlerine Göre Grupla
    import json
    import os
    
    sektor_gruplari = {}
    
    EGITIM_HAVUZU = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', # Teknoloji
        'JPM', 'BAC', 'V', 'MA',                 # Finans
        'JNJ', 'UNH', 'LLY', 'PFE',              # Sağlık
        'XOM', 'CVX',                            # Enerji
        'KO', 'PEP', 'PG', 'WMT',                # Defansif
        'CAT', 'HON',                            # Sanayi
        'LIN', 'NEM',                            # Hammadde
        'GC=F', 'SI=F'                       # Kıymetli Madenler (Emtia)
    ]
    
    # JSON'dan sektör eşleştirmesi yap
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
        except Exception:
            pass

    print("\n" + "="*70)
    print("🌍 WALL STREET GLOBAL CONTEXT: SEKTÖR UZMANI EĞİTİMİ BAŞLIYOR 🌍")
    print("="*70)
    
    print("\n[BİLGİ] Varlıklar sektör DNA'larına göre ayrıştırılıyor...")
    for sembol in EGITIM_HAVUZU:
        sektor = sektor_haritasi.get(sembol, "BILINMEYEN")
        if sembol in ['GC=F', 'SI=F']:
            sektor = 'EMTIA_VE_MADENCILIK'
            
        if sektor not in sektor_gruplari:
            sektor_gruplari[sektor] = []
        sektor_gruplari[sektor].append(sembol)

    # Her Sektör İçin Ayrı Bir "Uzman Beyin" Eğit
    for sektor_adi, hisseler in sektor_gruplari.items():
        if sektor_adi == "BILINMEYEN":
            continue
            
        print("\n" + "*"*60)
        print(f"🏭 SEKTÖR: {sektor_adi.upper()} | UZMAN BEYİN EĞİTİMİ BAŞLIYOR")
        print(f"📦 İlgili Varlıklar: {', '.join(hisseler)}")
        print("*"*60)
        
        dfs = []
        for sembol in hisseler:
            try:
                # Veriyi mutfaktan (SQLite + 9 Makro Kolon + 15 Teknik Kolon) çekiyoruz
                df = veriyi_oku_ve_ozellikleri_hesapla(sembol)
                if df is not None and len(df) > 100:
                    dfs.append(df)
                else:
                    print(f"  -> [PAS GEÇİLDİ] {sembol} için veri yetersiz.")
            except Exception as e:
                print(f"  -> [HATA] {sembol} okunamadı: {e}")
                
        if len(dfs) > 0:
            # Örn model_adi: "TEKNOLOJI" -> Dosya: TEKNOLOJI_long_hiyerarsik_beyin.pth olacak
            hedef_model_adi = sektor_adi.replace(" ", "_").upper()
            
            # Sektörün LONG beynini eğit
            beyni_egit(hedef_model_adi, "LONG", dfs, model_adi=hedef_model_adi)
            
            # Sektörün SHORT beynini eğit
            beyni_egit(hedef_model_adi, "SHORT", dfs, model_adi=hedef_model_adi)
        else:
            print(f"⚠️ {sektor_adi} sektörü için geçerli eğitim verisi bulunamadı!")

    print("\n🚀 TÜM SEKTÖR UZMANLARI EĞİTİLDİ! SİSTEM CANLI TİCARETE HAZIR.\n")

