import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as si
import time
import os
import sqlite3

# ==========================================
# 1. BÖLÜM: BLACK-SCHOLES YUNANLAR MOTORU
# ==========================================
class BlackScholesMotoru:
    @staticmethod
    def _d1_d2_hesapla(S, K, T, r, sigma):
        sigma = np.maximum(sigma, 1e-9)
        T = np.maximum(T, 1e-9)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def tum_yunanlari_hesapla(cls, S, K, T, r, sigma, opsiyon_tipi='call'):
        d1, d2 = cls._d1_d2_hesapla(S, K, T, r, sigma)
        pdf_d1 = si.norm.pdf(d1)
        cdf_d1 = si.norm.cdf(d1)
        cdf_d2 = si.norm.cdf(d2)
        cdf_neg_d1 = si.norm.cdf(-d1)
        cdf_neg_d2 = si.norm.cdf(-d2)
        
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = (S * pdf_d1 * np.sqrt(T)) / 100.0
        
        if opsiyon_tipi.lower() == 'call':
            delta = cdf_d1
            theta_hesap = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
            theta = theta_hesap / 365.0
            rho = (K * T * np.exp(-r * T) * cdf_d2) / 100.0
        else:
            delta = cdf_d1 - 1.0
            theta_hesap = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_neg_d2
            theta = theta_hesap / 365.0
            rho = -(K * T * np.exp(-r * T) * cdf_neg_d2) / 100.0
            
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}


# ==========================================
# 2. BÖLÜM: VOLATİLİTE KAPISI, PARQUET VE BALİNA HAFIZASI
# ==========================================
class OpsiyonGEXRadyosu:
    def __init__(self, risk_serbest_faiz=0.045):
        self.r = risk_serbest_faiz
        self.eski_spot_fiyatlar = {} # Volatilite kapısı ve Fiyat Eğilimi için
        self.eski_call_deltalar = {} # Balina Radarı (Kısa Süreli Hafıza)
        self.eski_gex_verileri = {}  # Ffill (İleri taşıma) için hafıza
        
        self.arsiv_klasoru = "opsiyon_parquet_arsivi"
        if not os.path.exists(self.arsiv_klasoru):
            os.makedirs(self.arsiv_klasoru)

    def volatilite_kapisi_acik_mi(self, sembol, anlik_fiyat):
        if sembol not in self.eski_spot_fiyatlar:
            self.eski_spot_fiyatlar[sembol] = anlik_fiyat
            return True 
            
        eski_fiyat = self.eski_spot_fiyatlar[sembol]
        
        if eski_fiyat == 0:
            self.eski_spot_fiyatlar[sembol] = anlik_fiyat
            return True
            
        degisim_yuzdesi = abs((anlik_fiyat - eski_fiyat) / eski_fiyat) * 100
        
        if degisim_yuzdesi >= 0.5:
            # Fiyat güncelleniyor, kapı açıldı
            return True
        return False

    def gex_duvarlarini_haritala(self, sembol, anlik_fiyat):
        if not self.volatilite_kapisi_acik_mi(sembol, anlik_fiyat):
            print(f"[VOLATİLİTE KAPISI] {sembol} hareketsiz. Önceki veriler (ffill) kullanılıyor.")
            return self.eski_gex_verileri.get(sembol, None)

        try:
            tkr = yf.Ticker(sembol)
            vadeler = tkr.options
            if not vadeler: return None
            
            en_yakin_vade = vadeler[0]
            T = max((pd.to_datetime(en_yakin_vade) - pd.to_datetime(time.strftime("%Y-%m-%d"))).days, 0.5) / 365.0
            
            zincir = tkr.option_chain(en_yakin_vade)
            calls = zincir.calls.copy().fillna(0) if not zincir.calls.empty else pd.DataFrame()
            puts = zincir.puts.copy().fillna(0) if not zincir.puts.empty else pd.DataFrame()
            
            # --- BLACK-SCHOLES HESAPLAMALARI ---
            if not calls.empty:
                calls['Gamma'] = calls.apply(lambda row: BlackScholesMotoru.tum_yunanlari_hesapla(anlik_fiyat, row['strike'], T, self.r, row.get('impliedVolatility', 0.0), 'call')['Gamma'], axis=1)
                calls['GEX'] = calls['Gamma'] * calls.get('openInterest', 0.0) * 100 * anlik_fiyat
                calls['Delta'] = calls.apply(lambda row: BlackScholesMotoru.tum_yunanlari_hesapla(anlik_fiyat, row['strike'], T, self.r, row.get('impliedVolatility', 0.0), 'call')['Delta'], axis=1)
            else:
                calls = pd.DataFrame(columns=['strike', 'GEX', 'openInterest', 'Delta', 'volume'])
                
            if not puts.empty:
                puts['Gamma'] = puts.apply(lambda row: BlackScholesMotoru.tum_yunanlari_hesapla(anlik_fiyat, row['strike'], T, self.r, row.get('impliedVolatility', 0.0), 'put')['Gamma'], axis=1)
                puts['GEX'] = puts['Gamma'] * puts.get('openInterest', 0.0) * 100 * anlik_fiyat * (-1)
            else:
                puts = pd.DataFrame(columns=['strike', 'GEX', 'openInterest', 'volume'])
            
            c_clean = calls[['strike', 'GEX', 'openInterest', 'Delta', 'volume']]
            p_clean = puts[['strike', 'GEX', 'openInterest', 'volume']]
            gex_df = pd.merge(c_clean, p_clean, on='strike', how='outer', suffixes=('_Call', '_Put')).fillna(0)
            
            # --- DUVARLAR VE REJİM ---
            gex_df['Net_GEX'] = gex_df['GEX_Call'] + gex_df['GEX_Put']
            put_wall = float(gex_df.loc[gex_df['openInterest_Put'].idxmax()]['strike']) if not gex_df.empty else 0
            call_wall = float(gex_df.loc[gex_df['openInterest_Call'].idxmax()]['strike']) if not gex_df.empty else 0
            gamma_flip = float(gex_df.loc[gex_df['Net_GEX'].abs().idxmin()]['strike']) if not gex_df.empty else 0
            
            # --- BALİNA RADARI (DELTA İVMESİ) HESAPLAMA ---
            toplam_call_delta = float(gex_df['Delta'].sum())
            eski_delta = self.eski_call_deltalar.get(sembol, None)
            delta_egilimi = 0.0 # Varsayılan (İlk 15 dakika)
            
            if eski_delta is not None and eski_delta > 0:
                # Delta ne kadar hızlandı? (Yüzde olarak)
                delta_egilimi = ((toplam_call_delta - eski_delta) / eski_delta) * 100
                
            self.eski_call_deltalar[sembol] = toplam_call_delta # Hafızayı güncelle
            
            hacim_pcr = float(gex_df['volume_Put'].sum() / gex_df['volume_Call'].sum()) if gex_df['volume_Call'].sum() > 0 else 1.0

            # --- PARQUET SIKIŞTIRMASI ---
            dosya_adi = f"{self.arsiv_klasoru}/{sembol}_{time.strftime('%Y%m%d_%H%M')}.parquet"
            gex_df.to_parquet(dosya_adi, engine='pyarrow', compression='snappy')

            sonuc = {
                'Spot_Fiyat': anlik_fiyat,
                'Put_Wall': put_wall,         
                'Call_Wall': call_wall,       
                'Gamma_Flip': gamma_flip,     
                'Delta_Egilimi': round(delta_egilimi, 2), # Balinanın Alım Hızı (%)
                'Hacim_PCR': round(hacim_pcr, 4),        
                'Vade': en_yakin_vade
            }
            
            self.eski_gex_verileri[sembol] = sonuc
            self.eski_spot_fiyatlar[sembol] = anlik_fiyat # Fiyatı güncelle
            return sonuc
            
        except Exception as e:
            print(f"❌ GEX Radarı Hatası: {e}")
            return self.eski_gex_verileri.get(sembol, None)


# ==========================================
# 3. BÖLÜM: DİNAMİK RİSK VE VİTES KUTUSU
# ==========================================
def dinamik_vites_hesapla(ai_sinyali, spot_fiyat, gamma_flip, fiyat_egilimi, opsiyon_delta_egilimi):
    risk_carpani = 1.0
    rejim_notu = "STANDART (Otoban)"

    negatif_gamma_rejimi = spot_fiyat < gamma_flip
    
    if ai_sinyali == "LONG" and negatif_gamma_rejimi:
        risk_carpani *= 0.25
        rejim_notu = "NEGATİF GAMMA (Buzlu Yol - Defansif %25 Lot)"
    elif ai_sinyali == "SHORT" and negatif_gamma_rejimi:
        risk_carpani *= 1.25
        rejim_notu = "NEGATİF GAMMA (Şelale Desteği - Agresif %125 Lot)"

    delta_uyumsuzlugu = (fiyat_egilimi < 0) and (opsiyon_delta_egilimi > 0)
    if ai_sinyali == "LONG" and delta_uyumsuzlugu:
        risk_carpani *= 1.50
        rejim_notu += " | 🐋 GİZLİ BALİNA ALIMI (Turbo %150 Lot)"
        
    return max(0.1, min(risk_carpani, 2.0)), rejim_notu

def lot_hesapla(kasa_bakiyesi, temel_risk_yuzdesi, stop_mesafesi, risk_carpani):
    baz_risk_dolari = kasa_bakiyesi * (temel_risk_yuzdesi / 100.0)
    nihai_risk_dolari = baz_risk_dolari * risk_carpani
    alinacak_lot = nihai_risk_dolari / max(stop_mesafesi, 0.01)
    return round(alinacak_lot, 2), nihai_risk_dolari


# ==========================================
# 4. BÖLÜM: KARA KUTU (HATA OTOPSİ GÜNLÜĞÜ)
# ==========================================
def kara_kutu_logla(sembol, spot_fiyat, put_wall, call_wall, gamma_flip, rejim_notu, ai_sinyali, ai_guven, risk_carpani, risk_dolari, lot, db_yolu="telemetri_kara_kutu.db"):
    try:
        baglanti = sqlite3.connect(db_yolu)
        cursor = baglanti.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operasyon_gunlugu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tarih TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sembol TEXT,
                spot_fiyat REAL,
                put_wall REAL,
                call_wall REAL,
                gamma_flip REAL,
                rejim TEXT,
                ai_sinyali TEXT,
                ai_guven REAL,
                risk_carpani REAL,
                risk_dolari REAL,
                alinan_lot REAL
            )
        """)
        
        cursor.execute("""
            INSERT INTO operasyon_gunlugu 
            (sembol, spot_fiyat, put_wall, call_wall, gamma_flip, rejim, ai_sinyali, ai_guven, risk_carpani, risk_dolari, alinan_lot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sembol, spot_fiyat, put_wall, call_wall, gamma_flip, rejim_notu, ai_sinyali, ai_guven, risk_carpani, risk_dolari, lot))
        
        baglanti.commit()
        baglanti.close()
    except Exception as e:
        print(f"❌ Kara Kutu Yazma Hatası: {e}")

# ==========================================
# 5. BÖLÜM: MASTER ENTEGRASYON VE TETİKLEYİCİ
# ==========================================
# Bu radyoyu dışarıdan erişilebilir statik bir obje olarak tutuyoruz ki 
# her işlemde hafıza (eski_deltalar) sıfırlanmasın.
GLOBAL_RADYO = OpsiyonGEXRadyosu()

def karar_ve_risk_motoru(sembol, mevcut_kasa, ai_sinyali, ai_guven):
    """
    proje2.py içinden yapay zeka sinyal ürettiğinde doğrudan bu fonksiyon çağrılır.
    """
    tkr = yf.Ticker(sembol)
    hist = tkr.history(period="1d")
    anlik_fiyat = float(hist['Close'].iloc[-1]) if not hist.empty else 0
    
    # 1. Hafızadan eski fiyatı alıp fiyatın ne yöne gittiğini bul (% olarak)
    eski_fiyat = GLOBAL_RADYO.eski_spot_fiyatlar.get(sembol, anlik_fiyat)
    fiyat_egilimi = ((anlik_fiyat - eski_fiyat) / eski_fiyat) * 100 if eski_fiyat > 0 else 0.0
    
    # 2. Opsiyon Radarını Çalıştır (Duvarları ve Delta İvmesini Çek)
    gex_ozet = GLOBAL_RADYO.gex_duvarlarini_haritala(sembol, anlik_fiyat)
    
    if not gex_ozet:
        return {"Aksiyon": "PAS", "Hata": "Veri çekilemedi, işlem pas geçildi."}
        
    spot = gex_ozet['Spot_Fiyat']
    g_flip = gex_ozet['Gamma_Flip']
    put_wall = gex_ozet['Put_Wall']
    call_wall = gex_ozet['Call_Wall']
    opsiyon_delta_egilimi = gex_ozet['Delta_Egilimi']
    
    # 3. Dinamik Vites (Risk Çarpanı) - BALİNA RADARI BURADA ÇALIŞIR
    risk_carpani, rejim_notu = dinamik_vites_hesapla(
        ai_sinyali, spot, g_flip, fiyat_egilimi, opsiyon_delta_egilimi
    )
    
    # 4. Kasa ve Lot Yönetimi (Örn: Kasadan maksimum %1 risk, 1.5 ATR Stop)
    # Gerçek sistemde stop mesafesini AI'nin verdiği ATR verisinden çekebilirsin.
    stop_mesafesi = 1.5 
    lot, dolarlik_risk = lot_hesapla(mevcut_kasa, 1.0, stop_mesafesi, risk_carpani)
    
    # 5. Otopsi İçin Kara Kutuya Yaz (SQLite)
    kara_kutu_logla(
        sembol, spot, put_wall, call_wall, g_flip, rejim_notu, 
        ai_sinyali, ai_guven, risk_carpani, dolarlik_risk, lot
    )
    
    # Eğer sistem risk çarpanını 0.1 gibi çok düşük bir yere çekerse, hiç komisyon ödememek için işlemi PAS geç.
    nihai_aksiyon = ai_sinyali if risk_carpani > 0.15 else "PAS"
    
    return {
        "Aksiyon": nihai_aksiyon,
        "Lot": lot,
        "Risk_Dolari": dolarlik_risk,
        "Rejim": rejim_notu,
        "GEX_Ozet": gex_ozet
    }

if __name__ == "__main__":
    print("Opsiyon GEX Motoru başlatılıyor... (Test Modu)")
    # Basit bir test
    test_sonuc = karar_ve_risk_motoru("AAPL", 10000.0, "LONG", 85.0)
    print("Test Sonucu:", test_sonuc)
