import pandas as pd
import sqlite3
import os

def analiz_yap():
    if not os.path.exists("sanal_islem_gecmisi.csv"):
        print("CSV bulunamadı!")
        return

    df = pd.read_csv("sanal_islem_gecmisi.csv")
    
    erken_kacis_islemleri = df[df['Sebep'].str.contains('AI Erken Kaçış')]
    toplam = len(erken_kacis_islemleri)
    print(f"Toplam Erken Kaçış İşlemi: {toplam}\n")
    
    if toplam == 0:
        return

    conn = sqlite3.connect("yapay_zeka_veritabani.sqlite")
    
    # Hızlı okuma için sembol verilerini cache'leyelim
    veriler = {}
    for sembol in erken_kacis_islemleri['Sembol'].unique():
        veriler[sembol] = pd.read_sql_query(f'SELECT Datetime, High, Low, Close FROM "{sembol}"', conn, index_col="Datetime")
        veriler[sembol].index = pd.to_datetime(veriler[sembol].index)

    dogru_karar = 0
    fake_yemis = 0
    
    kacilan_zarar_usd = 0
    kacirilan_kar_usd = 0
    
    for idx, row in erken_kacis_islemleri.iterrows():
        sembol = row['Sembol']
        cikis_tarihi = pd.to_datetime(row['Cikis_Tarihi'])
        giris_fiyati = float(row['Giris_Fiyati'])
        hedef = float(row['Hedef_Fiyati'])
        stop = float(row['Stop_Fiyati'])
        cikis_fiyati = float(row['Cikis_Fiyati'])
        adet = float(row['Adet'])
        yon = row['Yon']
        
        simdiki_pl = row['P&L_USD']
        
        # Gelecekteki veriler
        df_gelecek = veriler[sembol].loc[cikis_tarihi:]
        if len(df_gelecek) <= 1:
            continue # Veri bitmiş
            
        df_gelecek = df_gelecek.iloc[1:] # Çıkış yaptığımız günden sonrasına bakalım
        
        gercek_sonuc_fiyati = None
        hit_target = False
        hit_stop = False
        
        for index, gun_verisi in df_gelecek.iterrows():
            high = gun_verisi['High']
            low = gun_verisi['Low']
            
            if yon == "LONG":
                if low <= stop:
                    gercek_sonuc_fiyati = stop
                    hit_stop = True
                    break
                elif high >= hedef:
                    gercek_sonuc_fiyati = hedef
                    hit_target = True
                    break
            else: # SHORT
                if high >= stop:
                    gercek_sonuc_fiyati = stop
                    hit_stop = True
                    break
                elif low <= hedef:
                    gercek_sonuc_fiyati = hedef
                    hit_target = True
                    break
                    
        if not hit_target and not hit_stop:
            gercek_sonuc_fiyati = df_gelecek.iloc[-1]['Close'] # Son gün kapanış
            
        # Eğer kaçmasaydık P&L ne olacaktı?
        if yon == "LONG":
            orijinal_pl = (gercek_sonuc_fiyati - giris_fiyati) * adet
        else:
            orijinal_pl = (giris_fiyati - gercek_sonuc_fiyati) * adet
            
        fark_usd = simdiki_pl - orijinal_pl 
        
        # Kaçarak orijinal PL'ye göre ne yaptık?
        if fark_usd > 0:
            # Kaçmak mantıklıymış! Orijinal durum daha kötüydü. (Daha az zarar ettik veya orijinal zarardı biz kâr ettik)
            dogru_karar += 1
            kacilan_zarar_usd += fark_usd
        else:
            # Fake yemişiz! Orijinalde beklesek daha çok kazanacaktık.
            fake_yemis += 1
            kacirilan_kar_usd += abs(fark_usd)
            
    print(f"--> DOGRU KARAR (Iyi ki kacmisiz, Stop'a gidecekti veya zarari buyutecekti): {dogru_karar} islem")
    print(f"    -> AI sayesinde kurtarilan para: ${kacilan_zarar_usd:.2f}\n")
    
    print(f"--> FAKE YEMIS (Kacmamaliydik, Hedef'e gidecekti veya kar buyuyecekti): {fake_yemis} islem")
    print(f"    -> Erken kacip masada birakilan para: ${kacirilan_kar_usd:.2f}\n")

if __name__ == "__main__":
    analiz_yap()
