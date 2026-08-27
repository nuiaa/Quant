import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import *
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# ================= AYARLAR =================
PORT = 4002
KAC_AY_GERI = 240        # Gidilecek max ay sayisi (20 Yil)
BEKLEME_SURESI = 12      # Pacing cezasi yememek icin uyku suresi
ARSIV_KLASORU = "ibkr_1m_arsiv"
DURUM_DOSYASI = "indirme_durumu.json"
# ===========================================

if not os.path.exists(ARSIV_KLASORU):
    os.makedirs(ARSIV_KLASORU)

def load_durum():
    if os.path.exists(DURUM_DOSYASI):
        try:
            with open(DURUM_DOSYASI, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_durum(durum):
    with open(DURUM_DOSYASI, 'w', encoding='utf-8') as f:
        json.dump(durum, f, indent=4)

def get_symbols():
    with open('piyasa_haritasi.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    tickers = []
    for sec_v in d['SEKTORLER'].values():
        for ind_v in sec_v['Endustriler'].values():
            tickers.extend(ind_v['Hisseler'])
    
    # Altin (GC=F) ve Gumus (SI=F) IBKR'de vadeli kontrattir ve cekmesi cok zordur (Surekli bittiklerinden dolayi Roll edilir).
    # Bu yuzden %99 ayni veriyi iceren devasa ETF'leri (GLD ve SLV) kullaniyoruz.
    tickers.extend(['GLD', 'SLV'])
    
    return sorted(list(set(tickers))) # Tekrarlayan hisseleri temizler ve alfabetik siralar

async def gecmis_veri_cek(ib, sembol, bitis_tarihi, durationStr='1 M', barSizeSetting='1 min'):
    contract = Stock(sembol, 'SMART', 'USD')
    try:
        await ib.qualifyContractsAsync(contract)
    except Exception as e:
        print(f"[{sembol}] Gecersiz veya taninmayan sozlesme: {e}")
        return None
        
    print(f"[{sembol}] {bitis_tarihi.strftime('%Y-%m-%d %H:%M:%S')} oncesi 1 aylik veriler isteniyor...")
    
    try:
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime=bitis_tarihi,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
    except Exception as e:
        print(f"[{sembol}] IBKR Veri Hatasi: {e}")
        return None
        
    if not bars:
        return None
        
    df = util.df(bars)
    return df

async def main():
    ib = IB()
    try:
        import random
        # Pacing limitine cok takilmamasi icin clientId'yi rastgele verebiliriz
        rastgele_id = random.randint(100, 9999)
        await ib.connectAsync('127.0.0.1', PORT, clientId=rastgele_id)
        print(f"[BASARILI] IB Gateway'e baglanildi! (Client ID: {rastgele_id}) Otonom Madenci Baslatiliyor...")
    except Exception as e:
        print(f"[HATA] Gateway Baglantisi Kurulamadi: {e}")
        return
        
    semboller = get_symbols()
    durum = load_durum()
    
    # --- YENI KURAL: Yarim kalanlari basa al, hic baslamayanlari ortaya, bitenleri sona ---
    kismi_inenler = [s for s in semboller if s in durum and durum[s] != "TAMAMLANDI"]
    hic_baslamayanlar = [s for s in semboller if s not in durum]
    bitenler = [s for s in semboller if durum.get(s) == "TAMAMLANDI"]
    sirali_semboller = kismi_inenler + hic_baslamayanlar + bitenler
    
    print(f"Toplam {len(semboller)} varlik. ({len(kismi_inenler)} Yarim Kalan Oncelikli, {len(hic_baslamayanlar)} Bekleyen)\nArkaplanda calismaya devam edebilir, kapatsaniz da kaldigi yerden baslar.")
    
    for sembol in sirali_semboller:
        if durum.get(sembol) == "TAMAMLANDI":
            print(f"[ATLANDI] {sembol} zaten tamamen indirilmis. Siradaki hisseye geciliyor...")
            continue
            
        csv_yolu = os.path.join(ARSIV_KLASORU, f"{sembol}_1m.csv")
        
        # Eger indirme_durumu.json icinde yarim kalmis bir tarih varsa onu yukle
        if sembol in durum:
            try:
                hedef_tarih = datetime.strptime(durum[sembol], "%Y-%m-%d %H:%M:%S")
                print(f"\n[DEVAM EDILIYOR] {sembol} indirmesine kalinan tarihten ({hedef_tarih}) devam ediliyor.")
            except:
                hedef_tarih = datetime.now()
        else:
            print(f"\n[YENI BASLIYOR] {sembol} indirilmesine bugunden itibaren baslaniyor...")
            hedef_tarih = datetime.now()
            
        bos_donus_sayaci = 0
        ay_sayaci = 0
        
        while ay_sayaci < KAC_AY_GERI:
            ay_sayaci += 1
            df_parca = await gecmis_veri_cek(ib, sembol, hedef_tarih)
            
            if df_parca is not None and not df_parca.empty:
                bos_donus_sayaci = 0 # Basarili olursa sayaci sifirla
                
                en_eski_tarih = df_parca['date'].min()
                
                df_parca.set_index('date', inplace=True)
                # Sadece ise yarar 5 kolonu al
                if set(['open', 'high', 'low', 'close', 'volume']).issubset(df_parca.columns):
                    df_parca = df_parca[['open', 'high', 'low', 'close', 'volume']]
                
                # Dosya yoksa header yaz (ilk parca), varsa sadece append yap (veri ekle)
                header = not os.path.exists(csv_yolu)
                df_parca.to_csv(csv_yolu, mode='a', header=header)
                
                hedef_tarih = en_eski_tarih
                print(f"  -> {len(df_parca)} adet 1m mum kaydedildi. Yeni hedef tarih: {en_eski_tarih}")
                
                # Basarili islemi JSON'a kaydet (Elektrik gitse bile bu tarihten devam eder)
                durum[sembol] = hedef_tarih.strftime("%Y-%m-%d %H:%M:%S")
                save_durum(durum)
            else:
                bos_donus_sayaci += 1
                hedef_tarih = hedef_tarih - timedelta(days=30)
                print(f"  -> Veri donmedi, tarih 30 gun geriye kaydiriliyor. (Bos Donus Sayaci: {bos_donus_sayaci}/3)")
                
                # Eger 3 ay ust uste veri donmezse (Hissenin borsaya acilma tarihine ulasildi demektir)
                if bos_donus_sayaci >= 3:
                    print(f"[TAMAMLANDI] {sembol} icin IBKR'nin saglayabildigi en eski veriye ulasildi!")
                    break
                    
            print(f"  -> IBKR API Limitleri geregi {BEKLEME_SURESI} saniye bekleniyor...\n")
            await asyncio.sleep(BEKLEME_SURESI)
            
        # 240 ay biterse veya en eski veriye ulasilirsa TAMAMLANDI olarak etiketle
        # Veriler parca parca eklendigi icin dosyadaki siralamayi (kronolojik) duzelt ve oyle tamamla
        try:
            if os.path.exists(csv_yolu):
                print(f"[DUZENLEME] {sembol} dosyasindaki tarihler kronolojik olarak siralaniyor...")
                df_temiz = pd.read_csv(csv_yolu, parse_dates=['date'])
                df_temiz.sort_values('date', inplace=True)
                df_temiz.drop_duplicates(subset=['date'], inplace=True)
                df_temiz.to_csv(csv_yolu, index=False)
                durum[sembol] = "TAMAMLANDI"
            else:
                print(f"[UYARI] {sembol} icin hic veri indirilemedi. Bir sonraki calistirmada tekrar denenecek.")
                durum.pop(sembol, None)
        except Exception as e:
            print(f"[UYARI] {sembol} siralama hatasi: {e}")
            if os.path.exists(csv_yolu):
                durum[sembol] = "TAMAMLANDI"
            else:
                durum.pop(sembol, None)
            
        save_durum(durum)
        
    print("\n[MUKEMMEL] Otonom Veri Madencisi (Crawler) tum 517 varlik icin isini basariyla bitirdi!")
    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
