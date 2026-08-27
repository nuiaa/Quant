import os
import json
import time
import datetime
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ARSIV_KLASORU = "opsiyon_arsivi"
KATALOG_DOSYASI = "opsiyon_katalog.json"
MAX_WORKERS = 10  # yfinance API limitlerine takılmamak için

def katalog_yukle():
    if os.path.exists(KATALOG_DOSYASI):
        with open(KATALOG_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def katalog_kaydet(katalog):
    with open(KATALOG_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(katalog, f, indent=4)

def gecmis_bosluklari_doldur(katalog, bugun_str):
    """
    Eğer katalog boşsa, bugünden başlar.
    Eğer katalogda eski bir tarih varsa ve bugünle arası boşsa, o günlere 'NO DATA' basar.
    """
    if not katalog:
        return
        
    tarihler = sorted([t for t in katalog.keys()])
    if not tarihler:
        return
        
    son_tarih_str = tarihler[-1]
    son_tarih = datetime.datetime.strptime(son_tarih_str, "%Y-%m-%d").date()
    bugun = datetime.datetime.strptime(bugun_str, "%Y-%m-%d").date()
    
    fark = (bugun - son_tarih).days
    
    # 1 günden fazla fark varsa, aradaki günleri NO DATA olarak işaretle
    if fark > 1:
        for i in range(1, fark):
            aradaki_gun = son_tarih + datetime.timedelta(days=i)
            # Sadece hafta içiyse (Pzt-Cuma) NO DATA bas
            if aradaki_gun.weekday() < 5:
                aradaki_gun_str = aradaki_gun.strftime("%Y-%m-%d")
                if aradaki_gun_str not in katalog:
                    katalog[aradaki_gun_str] = "NO DATA"
                    print(f"[KATALOG] Bot kapalı kaldığı için {aradaki_gun_str} 'NO DATA' olarak işaretlendi.")

def klasor_olustur(tarih_str):
    tarih_obj = datetime.datetime.strptime(tarih_str, "%Y-%m-%d")
    hedef_yol = os.path.join(ARSIV_KLASORU, str(tarih_obj.year), f"{tarih_obj.month:02d}", f"{tarih_obj.day:02d}")
    if not os.path.exists(hedef_yol):
        os.makedirs(hedef_yol)
    return hedef_yol

def sembol_opsiyon_indir(sembol, hedef_klasor):
    try:
        tkr = yf.Ticker(sembol)
        vadeler = tkr.options
        
        if not vadeler:
            return sembol, False, "Opsiyon zinciri boş"
            
        # Sadece en yakın 2 vadeyi kaydedelim (Genelde likidite ve gex burada birikir)
        # Çok ileriki vadeler diski şişirebilir.
        kaydedilen_vade_sayisi = 0
        
        for vade in vadeler[:2]:
            zincir = tkr.option_chain(vade)
            calls = zincir.calls
            puts = zincir.puts
            
            # Sadece gerekli kolonları al (Greeks'leri sildik, %80 küçüldü)
            istenen_kolonlar = ['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']
            
            # Eğer DataFrame boş değilse filtrele
            if not calls.empty:
                mevcut_kolonlar = [c for c in istenen_kolonlar if c in calls.columns]
                c_clean = calls[mevcut_kolonlar].copy()
                c_clean['type'] = 'call'
                c_clean['vade'] = vade
            else:
                c_clean = pd.DataFrame()
                
            if not puts.empty:
                mevcut_kolonlar = [c for c in istenen_kolonlar if c in puts.columns]
                p_clean = puts[mevcut_kolonlar].copy()
                p_clean['type'] = 'put'
                p_clean['vade'] = vade
            else:
                p_clean = pd.DataFrame()
                
            birlesik = pd.concat([c_clean, p_clean], ignore_index=True)
            
            if not birlesik.empty:
                # NaN değerleri temizle (disk tasarrufu)
                birlesik.fillna(0, inplace=True)
                
                dosya_adi = os.path.join(hedef_klasor, f"{sembol}_{vade.replace('-', '')}.parquet")
                
                # brotli algoritması arşivleme için en iyi sıkıştırmayı sağlar
                birlesik.to_parquet(dosya_adi, engine='pyarrow', compression='brotli')
                kaydedilen_vade_sayisi += 1
                
        if kaydedilen_vade_sayisi > 0:
            return sembol, True, "OK"
        else:
            return sembol, False, "Geçerli opsiyon verisi bulunamadı"
            
    except Exception as e:
        return sembol, False, str(e)

def ambar_guncelle(sembol_listesi):
    bugun_str = datetime.datetime.now().strftime("%Y-%m-%d")
    katalog = katalog_yukle()
    
    # Geçmiş boşlukları kontrol et ve NO DATA bas
    gecmis_bosluklari_doldur(katalog, bugun_str)
    
    # Bugün zaten SUCCESS ise çık
    if katalog.get(bugun_str) == "SUCCESS":
        print(f"[{bugun_str}] Opsiyon verileri bugün zaten başarıyla indirilmiş. (SUCCESS)")
        return
        
    print(f"\n[{bugun_str}] 📦 OPSIYON VERI AMBARI ÇALIŞIYOR...")
    print(f"Hedeflenen sembol sayısı: {len(sembol_listesi)}")
    
    hedef_klasor = klasor_olustur(bugun_str)
    basarili_sayisi = 0
    hata_sayisi = 0
    
    # ThreadPool ile hızlıca indir
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(sembol_opsiyon_indir, s, hedef_klasor): s for s in sembol_listesi}
        
        for future in as_completed(future_to_symbol):
            sembol = future_to_symbol[future]
            try:
                sym, basari, mesaj = future.result()
                if basari:
                    basarili_sayisi += 1
                    print(f"   [+] {sym} -> Kaydedildi.")
                else:
                    hata_sayisi += 1
                    # print(f"   [-] {sym} -> Hata: {mesaj}") # Terminali çok kirletmemek için gizleyebiliriz
            except Exception as exc:
                hata_sayisi += 1
                
    print(f"\n[ÖZET] Başarılı: {basarili_sayisi} | Başarısız: {hata_sayisi}")
    
    # Eğer yeterince hisse başarılıysa günü SUCCESS işaretle
    if basarili_sayisi > (len(sembol_listesi) * 0.1): 
        katalog[bugun_str] = "SUCCESS"
    else:
        katalog[bugun_str] = "NO DATA"
        print("[UYARI] Çok az veri indirilebildiği için bugün 'NO DATA' olarak işaretlendi. Piyasalar kapalı olabilir.")
        
    katalog_kaydet(katalog)
    print("Katalog güncellendi.\n")

if __name__ == "__main__":
    # Test için proje2'deki hisse havuzunu veya piyasa_haritasi.json'ı okuyabiliriz.
    # Burada basitçe bir test listesi veriyorum. Canlıda proje2'den çağırılacak.
    test_listesi = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    ambar_guncelle(test_listesi)
