import os
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Database name
VERITABANI_ADI = "yapay_zeka_veritabani.sqlite"

# ==========================================
# AFML MODÜLLERİ (Hacim Çubukları & Kesirli Fark)
# ==========================================
def hacim_cubuklari_olustur(df_1m, hacim_esigi=50000):
    """
    AFML Standartlarında Zaman Çubuklarını Hacim Çubuklarına Çevirir.
    Piyasa durgunken saatlerce mum kapanmaz, haber düştüğünde dakikada 5 mum kapanır.
    """
    print(f"[AFML] {len(df_1m)} adet veriden {hacim_esigi} lotluk Hacim Çubukları (Volume Bars) presleniyor...")
    
    kumbara_hacim = 0
    cubuklar = []
    mum_baslangic_idx = 0
    
    for i in range(len(df_1m)):
        kumbara_hacim += df_1m['Volume'].iloc[i]
        
        # Kumbaradaki hacim, belirlediğimiz eşiğe ulaştıysa MUMU KAPAT!
        if kumbara_hacim >= hacim_esigi:
            kesit = df_1m.iloc[mum_baslangic_idx : i+1]
            
            yeni_mum = {
                'Datetime': df_1m.index[i],
                'Open': kesit['Open'].iloc[0],
                'High': kesit['High'].max(),
                'Low': kesit['Low'].min(),
                'Close': kesit['Close'].iloc[-1],
                'Volume': kumbara_hacim
            }
            cubuklar.append(yeni_mum)
            
            kumbara_hacim = 0
            mum_baslangic_idx = i + 1
            
    df_hacim = pd.DataFrame(cubuklar).set_index('Datetime')
    print(f"[AFML] Presleme tamam! {len(df_hacim)} adet Hacim Çubuğu elde edildi.")
    return df_hacim

def kesirli_agirlik_hesapla(d, pencere_boyutu):
    """Kesirli fark alma işlemi için Binom serisi ağırlıklarını üretir."""
    w = [1.0]
    for k in range(1, pencere_boyutu):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1])

def kesirli_fark_al(seri, d=0.4, pencere=60):
    """
    AFML Fractional Differentiation: Fiyatın hafızasını silmeden durağanlaştırır.
    d: Fark derecesi (0.4 genelde hisse senetleri için optimumdur)
    """
    agirliklar = kesirli_agirlik_hesapla(d, pencere)
    sonuc = np.full_like(seri, np.nan, dtype=float)
    
    seri_degerleri = seri.values
    for i in range(pencere, len(seri_degerleri)):
        sonuc[i] = np.dot(agirliklar, seri_degerleri[i-pencere : i])
        
    return pd.Series(sonuc, index=seri.index)

def opsiyon_metriklerini_cek(sembol):
    """
    yfinance üzerinden hissenin en yakın vadeli opsiyon zincirini çeker.
    """
    if "USD" in sembol or "-" in sembol:
        return 1.0, 0.0 
    try:
        tkr = yf.Ticker(sembol)
        vadeler = tkr.options
        if not vadeler or len(vadeler) == 0:
            return 1.0, 0.0
        en_yakin_vade = vadeler[0]
        zincir = tkr.option_chain(en_yakin_vade)
        calls, puts = zincir.calls, zincir.puts
        if 'openInterest' not in calls.columns or 'openInterest' not in puts.columns:
            return 1.0, 0.0
        call_oi_toplam = calls['openInterest'].fillna(0).sum()
        put_oi_toplam = puts['openInterest'].fillna(0).sum()
        pcr = float(put_oi_toplam / call_oi_toplam) if call_oi_toplam != 0 else 1.0
        net_opsiyon_gucu = float(call_oi_toplam - put_oi_toplam)
        net_opsiyon_gucu_norm = np.log1p(net_opsiyon_gucu) if net_opsiyon_gucu > 0 else (-np.log1p(abs(net_opsiyon_gucu)) if net_opsiyon_gucu < 0 else 0.0)
        return round(pcr, 4), round(net_opsiyon_gucu_norm, 4)
    except Exception:
        return 1.0, 0.0

class CustomMinMaxScaler:
    """
    scikit-learn kütüphanesi yüklü olmayan sistemler için pure NumPy/Pandas ile
    özellikleri belirtilen aralığa (varsayılan -1 ile 1) ölçekleyen sınıf.
    """
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit_transform(self, X):
        self.min_val = X.min(axis=0)
        self.max_val = X.max(axis=0)
        aralik = self.max_val - self.min_val
        aralik[aralik == 0] = 1e-8
        X_std = (X - self.min_val) / aralik
        scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            return X
        aralik = self.max_val - self.min_val
        aralik[aralik == 0] = 1e-8
        X_std = (X - self.min_val) / aralik
        scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

def veriyi_oku_ve_ozellikleri_hesapla(sembol, tablo_adi=None, db_yolu=VERITABANI_ADI, kesme_zamani=None, canli_mod=False, backtest_10y=False):
    """
    Veriyi okur, özelliklerini hesaplar ve modeli besler.
    canli_mod=True ise Hacim Çubuklarını (Volume Bars) kullanır.
    canli_mod=False ise standart Zaman Çubuklarını (günlük/15m) kullanır.
    backtest_10y=True ise SQLite yerine yfinance'ten 10 yıllık '1d' veri çeker.
    """
    df = pd.DataFrame()
    t_name = tablo_adi if tablo_adi else sembol
    
    # 1. Veri Çekme (Canlı Mod vs. Backtest/Gece Taraması)
    if backtest_10y:
        print(f"[{sembol}] 10 YILLIK BACKTEST MODU: Yahoo Finance'ten 10 yıllık 1D veri indiriliyor...")
        try:
            df = yf.download(sembol, period="10y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.strip().capitalize() for col in df.columns]
        except Exception as e:
            print(f"[HATA] {sembol} verisi çekilemedi: {e}")
            return None
    elif canli_mod:
        print(f"[{sembol}] CANLI MOD AKTİF: 1 Dakikalık veriler Hacim Çubuklarına dönüştürülüyor...")
        try:
            df_1m = yf.download(sembol, period="7d", interval="1m", progress=False)
            if isinstance(df_1m.columns, pd.MultiIndex):
                df_1m.columns = df_1m.columns.droplevel(1)
            df_1m.columns = [col.strip().capitalize() for col in df_1m.columns]
            
            df_1d = yf.download(sembol, period="30d", interval="1d", progress=False)
            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = df_1d.columns.droplevel(1)
            gunluk_ortalama_hacim = df_1d['Volume'].mean() if not df_1d.empty else 1000000
            
            # Dinamik hacim eşiği (Günlük hacmin 390 dakikaya bölümü: 1 muma düşen ortalama lot)
            dinamik_hacim_esigi = int(gunluk_ortalama_hacim / 390)
            if dinamik_hacim_esigi < 1000: 
                dinamik_hacim_esigi = 1000
                
            df = hacim_cubuklari_olustur(df_1m, hacim_esigi=dinamik_hacim_esigi)
        except Exception as e:
            print(f"[UYARI] {sembol} 1m canlı verisi çekilemedi. Veritabanına (Zaman Çubuklarına) dönülüyor. Hata: {e}")
            canli_mod = False
            
    if not canli_mod and not backtest_10y:
        if os.path.exists(db_yolu):
            try:
                with sqlite3.connect(db_yolu) as conn:
                    query = f'SELECT * FROM "{t_name}"'
                    df = pd.read_sql_query(query, conn, index_col='Datetime')
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    if kesme_zamani is not None:
                        df = df[df.index <= kesme_zamani]
            except Exception as e:
                print(f"[UYARI] Veritabanindan veri okunamadi: {e}")
    
    if df.empty or len(df) < 100:
        tarihler = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='15min')
        np.random.seed(42)
        fiyatlar = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.0002, 1000)))
        df = pd.DataFrame({'Datetime': tarihler.strftime('%Y-%m-%d %H:%M:%S'), 'Close': fiyatlar})
        df['Open'] = df['Close'].shift(1).fillna(100.0)
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.0003, 1000)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.0003, 1000)))
        df['Volume'] = np.random.randint(100, 5000, size=1000).astype(float)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

    df_vix = pd.DataFrame()
    df_dxy = pd.DataFrame()
    sektor_etfleri = {'XLK': 'RS_Teknoloji', 'XLF': 'RS_Finans', 'XLV': 'RS_Saglik', 'XLE': 'RS_Enerji', 'XLI': 'RS_Sanayi', 'XLP': 'RS_Defansif', 'XLB': 'RS_Hammadde'}
    sektor_verileri = {}

    if os.path.exists(db_yolu):
        try:
            with sqlite3.connect(db_yolu) as conn:
                try: 
                    df_vix = pd.read_sql_query('SELECT Datetime, Close as VIX FROM "cache__VIX_1d"', conn, index_col='Datetime')
                    df_vix.index = pd.to_datetime(df_vix.index)
                except Exception: pass
                try: 
                    df_dxy = pd.read_sql_query('SELECT Datetime, Close as DXY FROM "cache_DX_Y_NYB_1d"', conn, index_col='Datetime')
                    df_dxy.index = pd.to_datetime(df_dxy.index)
                except Exception: pass
                for etf, kolon_adi in sektor_etfleri.items():
                    try:
                        df_sektor = pd.read_sql_query(f'SELECT Datetime, Close FROM "cache_{etf}_1d"', conn, index_col='Datetime')
                        df_sektor.index = pd.to_datetime(df_sektor.index)
                        sektor_verileri[kolon_adi] = df_sektor['Close'].pct_change(periods=5) * 100.0
                    except Exception: pass
        except Exception: pass

    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.columns = [col.strip().capitalize() for col in df.columns]
    if not df_vix.empty: df = df.join(df_vix, how='left')
    else: df['VIX'] = np.random.uniform(12.0, 35.0, len(df))
    if not df_dxy.empty: df = df.join(df_dxy, how='left')
    else: df['DXY'] = np.random.uniform(98.0, 108.0, len(df))
    for kolon_adi in sektor_etfleri.values():
        if kolon_adi in sektor_verileri:
            df = df.join(sektor_verileri[kolon_adi].rename(kolon_adi).fillna(0.0), how='left')
        else:
            df[kolon_adi] = 0.0
    
    makro_kolonlar = ['VIX', 'DXY'] + list(sektor_etfleri.values())
    df[makro_kolonlar] = df[makro_kolonlar].ffill().fillna(0.0)

    df['Gun_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['Gun_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    df['Ay_Sin'] = np.sin(2 * np.pi * df.index.day / 31.0)
    df['Ay_Cos'] = np.cos(2 * np.pi * df.index.day / 31.0)
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Fiyat_EMA20_Farki'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['Fiyat_EMA200_Farki'] = (df['Close'] - df['EMA_200']) / df['EMA_200']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 0.00001))))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Sinyal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Sinyal']

    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Yuzde'] = df['ATR'] / df['Close']
    
    # ADX Hesaplaması (14 periyotluk Wilder's Smoothing)
    up = df['High'] - df['High'].shift(1)
    down = df['Low'].shift(1) - df['Low']
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.00001)
    df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

    df['BB_Orta'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Pozisyon'] = (df['Close'] - (df['BB_Orta'] - (bb_std * 2))) / ((df['BB_Orta'] + (bb_std * 2)) - (df['BB_Orta'] - (bb_std * 2)) + 0.00001)

    df['Ust_Fitil_Gucu'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / (df['ATR'] + 0.00001)
    df['Alt_Fitil_Gucu'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / (df['ATR'] + 0.00001)
    df['Hacim_Patlamasi_Orani'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 0.00001)
    df['Fiyat_Degisimi_5G'] = df['Close'].pct_change(periods=5)
    df['RSI_Degisimi_5G'] = df['RSI'].diff(periods=5) / 100.0
    df['Uyumsuzluk_Skoru'] = df['RSI_Degisimi_5G'] - df['Fiyat_Degisimi_5G']
    
    sarkma = (df['Low'].shift(1).rolling(10).min() - df['Low'])
    df['Likidite_Avi_Siddeti'] = sarkma.where((df['Low'] < df['Low'].shift(1).rolling(10).min()) & (df['Close'] > df['Low'].shift(1).rolling(10).min()), 0) / (df['ATR'] + 0.00001)
    df['BB_Sikisma_Orani'] = ((df['BB_Orta'] + (bb_std * 2)) - (df['BB_Orta'] - (bb_std * 2))) / (df['BB_Orta'] + 0.00001)
    
    df['HA_Govde_Gucu'] = (((df['Open']+df['High']+df['Low']+df['Close'])/4) - ((df['Open'].shift(1)+df['Close'].shift(1))/2)) / (((df['Open'].shift(1)+df['Close'].shift(1))/2) + 0.00001) * 100
    df['Rolling_VWAP_20'] = ((df['High']+df['Low']+df['Close'])/3 * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 0.00001)
    df['VWAP_Uzaklik'] = (df['Close'] - df['Rolling_VWAP_20']) / (df['Rolling_VWAP_20'] + 0.00001) * 100
    df['VWAP_Egilim_5G'] = df['Rolling_VWAP_20'].pct_change(periods=5) * 100
    
    # ==========================================
    # 4. ÇOKLU ZAMAN DİLİMİ (MTF) FIBONACCI RADARI
    # ==========================================
    
    # --- A) KISA VADE (60 GÜN / ~3 AY) ---
    df['Swing_High_60'] = df['High'].rolling(window=60).max()
    df['Swing_Low_60'] = df['Low'].rolling(window=60).min()
    Fark_60 = df['Swing_High_60'] - df['Swing_Low_60']
    
    df['Fib_382_Kisa'] = df['Swing_High_60'] - (Fark_60 * 0.382)
    df['Fib_618_Kisa'] = df['Swing_High_60'] - (Fark_60 * 0.618)
    
    # Kısa vadeli destek/direnç hatlarına % uzaklık
    df['Fib_382_K_Uzaklik'] = (df['Close'] - df['Fib_382_Kisa']) / (df['Fib_382_Kisa'] + 0.00001) * 100
    df['Fib_618_K_Uzaklik'] = (df['Close'] - df['Fib_618_Kisa']) / (df['Fib_618_Kisa'] + 0.00001) * 100

    # --- B) UZUN VADE (252 GÜN / 1 YIL) - KURUMSAL ÇAPA ---
    df['Swing_High_252'] = df['High'].rolling(window=252).max()
    df['Swing_Low_252'] = df['Low'].rolling(window=252).min()
    Fark_252 = df['Swing_High_252'] - df['Swing_Low_252']
    
    df['Fib_382_Uzun'] = df['Swing_High_252'] - (Fark_252 * 0.382)
    df['Fib_618_Uzun'] = df['Swing_High_252'] - (Fark_252 * 0.618)
    
    # Uzun vadeli kurumsal destek/direnç hatlarına % uzaklık
    df['Fib_382_U_Uzaklik'] = (df['Close'] - df['Fib_382_Uzun']) / (df['Fib_382_Uzun'] + 0.00001) * 100
    df['Fib_618_U_Uzaklik'] = (df['Close'] - df['Fib_618_Uzun']) / (df['Fib_618_Uzun'] + 0.00001) * 100
    
    # ==========================================
    # 5. KESİRLİ FARK ALMA (Fractional Differentiation)
    # ==========================================
    df['Frac_Close'] = kesirli_fark_al(df['Close'], d=0.4)
    
    # NaN Temizliği (Güvenlik Kalkanı)
    fib_kolonlari = ['Fib_382_K_Uzaklik', 'Fib_618_K_Uzaklik', 'Fib_382_U_Uzaklik', 'Fib_618_U_Uzaklik']
    df[fib_kolonlari] = df[fib_kolonlari].fillna(0)
    df.fillna(0, inplace=True)

    # ==========================================
    # AFML TRIPLE-BARRIER (ÜÇLÜ BARİYER) METODU
    # ==========================================
    ZAMAN_BARIYERI = 15     # Dikey Bariyer (İşlemin ömrü maksimum 15 mum)
    PT_CARPANI = 1.5        # Üst Bariyer (Kâr Al - Volatiliteye dinamik)
    SL_CARPANI = 1.5        # Alt Bariyer (Zarar Kes - Volatiliteye dinamik)
    
    hedef_listesi_long = np.zeros(len(df))
    hedef_listesi_short = np.zeros(len(df))
    
    # Sıfıra bölme hatalarını önle
    df['ATR'] = df['ATR'].replace(0, 0.01)
    
    for i in range(len(df) - ZAMAN_BARIYERI):
        giris_fiyati = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        # 1. Dinamik Bariyerlerin Çizilmesi (Volatility Scaling)
        ust_bariyer = giris_fiyati + (atr * PT_CARPANI)
        alt_bariyer = giris_fiyati - (atr * SL_CARPANI)
        
        gelecek_pencere = df.iloc[i+1 : i+1+ZAMAN_BARIYERI]
        
        long_basarili = 0  # 0 = Stop oldu veya Zaman Aşımına uğradı (Başarısız)
        short_basarili = 0
        
        # 2. LONG İÇİN KRONOLOJİK YOL İZLEME (Path Dependency)
        for j in range(len(gelecek_pencere)):
            mum = gelecek_pencere.iloc[j]
            # Fiyat önce alt bariyere (Stop) değdiyse işlem ölür
            if mum['Low'] <= alt_bariyer:
                break 
            # Fiyat önce üst bariyere (Hedef) değdiyse başarılı sayılır
            elif mum['High'] >= ust_bariyer:
                long_basarili = 1
                break
                
        # 3. SHORT İÇİN KRONOLOJİK YOL İZLEME
        for j in range(len(gelecek_pencere)):
            mum = gelecek_pencere.iloc[j]
            # Short için üst bariyer STOP'tur
            if mum['High'] >= ust_bariyer:
                break
            # Short için alt bariyer HEDEF'tir
            elif mum['Low'] <= alt_bariyer:
                short_basarili = 1
                break
                
        hedef_listesi_long[i] = long_basarili
        hedef_listesi_short[i] = short_basarili
        
    df['Hedef_Yonu_Long'] = hedef_listesi_long
    df['Hedef_Yonu_Short'] = hedef_listesi_short
    
    pcr, ops_gucu = opsiyon_metriklerini_cek(sembol)
    df['PCR_Seviyesi'], df['Net_Opsiyon_Gucu'] = pcr, ops_gucu
    
    df = piyasa_mikro_yapi_ozellikleri_ekle(df)
    return df.dropna()

def piyasa_mikro_yapi_ozellikleri_ekle(df):
    epsilon = 1e-8
    mum_boyu = df['High'] - df['Low']
    mum_boyu = mum_boyu.replace(0, epsilon) 

    alt_sinir = df[['Open', 'Close']].min(axis=1)
    df['Alt_Golge_Boyu'] = alt_sinir - df['Low']
    df['Alt_Golge_Orani'] = df['Alt_Golge_Boyu'] / mum_boyu

    ust_sinir = df[['Open', 'Close']].max(axis=1)
    df['Ust_Golge_Boyu'] = df['High'] - ust_sinir
    df['Ust_Golge_Orani'] = df['Ust_Golge_Boyu'] / mum_boyu

    ort_hacim_20 = df['Volume'].rolling(window=20).mean()
    df['Hacim_Anomalisi'] = df['Volume'] / (ort_hacim_20 + epsilon)

    fiyat_hh_14 = df['Close'].rolling(window=14).max().shift(1)
    fiyat_ll_14 = df['Close'].rolling(window=14).min().shift(1)
    rsi_hh_14 = df['RSI'].rolling(window=14).max().shift(1)
    rsi_ll_14 = df['RSI'].rolling(window=14).min().shift(1)

    df['Ayi_Uyumsuzlugu'] = np.where((df['Close'] > fiyat_hh_14) & (df['RSI'] < rsi_hh_14), 1, 0)
    df['Boga_Uyumsuzlugu'] = np.where((df['Close'] < fiyat_ll_14) & (df['RSI'] > rsi_ll_14), 1, 0)
    
    df.drop(['Alt_Golge_Boyu', 'Ust_Golge_Boyu'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df

def tensore_donustur(df_or_list, islem_yonu="LONG", val_oran=0.15, undersample=False):
    """
    İşlenmiş veriyi PyTorch Tensörlerine çevirir ve Sınıf Dengesizliğini (Class Imbalance) çözer.
    Zaman serisi sızıntısını engellemek için undersampling öncesi kronolojik Train-Val split yapar.
    df_or_list: Tek bir DataFrame veya DataFrame listesi (çoklu varlık eğitimi için)
    islem_yonu: "LONG" veya "SHORT"
    val_oran: Doğrulama seti oranı (varsayılan %15)
    undersample: Sınıf dengesi için alt-örnekleme yapılıp yapılmayacağı (varsayılan False)
    """
    import random
    print(f"\n[{islem_yonu} BEYNİ İÇİN] Veriler Yapay Zeka Tensörlerine dönüştürülüyor...")
    
    # Modele vereceğimiz özelliklerin (Features) NİHAİ listesi (VIX ve DXY dahil 17 özellik)
    ozellikler = [
        'Fiyat_EMA20_Farki', 'Fiyat_EMA200_Farki', 'RSI', 
        'MACD', 'MACD_Histogram', 'ATR_Yuzde', 'BB_Pozisyon', 
        'Ust_Fitil_Gucu', 'Alt_Fitil_Gucu',       
        'Hacim_Patlamasi_Orani',                  
        'Fiyat_Degisimi_5G', 'RSI_Degisimi_5G',
        'Uyumsuzluk_Skoru', 'Likidite_Avi_Siddeti', 'BB_Sikisma_Orani',
        'VWAP_Egilim_5G',
        'VIX', 'DXY', 'PCR_Seviyesi', 'Net_Opsiyon_Gucu',
        'Fib_382_K_Uzaklik', 'Fib_618_K_Uzaklik',
        'Fib_382_U_Uzaklik', 'Fib_618_U_Uzaklik'
    ]
    
    # Girdi tipine göre liste haline getir
    if isinstance(df_or_list, list):
        dfs_list = df_or_list
    elif isinstance(df_or_list, tuple):
        dfs_list = list(df_or_list)
    else:
        dfs_list = [df_or_list]
        
    # Tüm özellik değerlerini toplayarak global CustomMinMaxScaler fit et
    scaler = CustomMinMaxScaler(feature_range=(-1, 1))
    
    all_features_list = []
    for df in dfs_list:
        if not df.empty:
            all_features_list.append(df[ozellikler].values)
            
    if not all_features_list:
        raise ValueError("HATA: Boş veri listesi veya yetersiz veri!")
        
    all_features = np.concatenate(all_features_list, axis=0)
    scaler.fit_transform(all_features)  # sets min_val and max_val globally
    
    X_train_raw, Y_train_raw = [], []
    X_val_raw, Y_val_raw = [], []
    
    GECMIS_MUM_SAYISI = 60
    
    # Her hisseyi kendi içinde kayan pencereye böl ve split et (boundary leakage önleme)
    for df in dfs_list:
        if df.empty or len(df) < GECMIS_MUM_SAYISI + 1:
            continue
            
        X_ham = df[ozellikler].values
        
        # Hangi beyni eğitiyorsak onun hedeflerini (Y) alıyoruz
        if islem_yonu == "LONG":
            Y_ham = df['Hedef_Yonu_Long'].values
        else:
            Y_ham = df['Hedef_Yonu_Short'].values
            
        # Global scaler ile ölçekle
        X_olcekli = scaler.transform(X_ham)
        
        X_stock, Y_stock = [], []
        for i in range(len(X_olcekli) - GECMIS_MUM_SAYISI):
            pencere = X_olcekli[i : (i + GECMIS_MUM_SAYISI)]
            hedef = Y_ham[i + GECMIS_MUM_SAYISI]
            X_stock.append(pencere)
            Y_stock.append(hedef)
            
        n_seq = len(X_stock)
        if n_seq == 0:
            continue
            
        # Kronolojik Train-Val split
        n_train = int(n_seq * (1.0 - val_oran))
        
        X_train_raw.extend(X_stock[:n_train])
        Y_train_raw.extend(Y_stock[:n_train])
        
        X_val_raw.extend(X_stock[n_train:])
        Y_val_raw.extend(Y_stock[n_train:])
        
    # ==========================================
    # SINIF DENGESİZLİĞİ ÇÖZÜMÜ (YALNIZCA TRAIN SETINE)
    # ==========================================
    basarili_indeksler = [i for i, y in enumerate(Y_train_raw) if y == 1]
    basarisiz_indeksler = [i for i, y in enumerate(Y_train_raw) if y == 0]
    
    print(f"Train Dengeleme Öncesi -> Başarılı (1): {len(basarili_indeksler)}, Başarısız (0): {len(basarisiz_indeksler)}")
    
    # Train setini dengele veya sadece karıştır
    if undersample:
        if len(basarili_indeksler) > 0 and len(basarisiz_indeksler) > 0:
            secilen_basarisizlar = random.sample(basarisiz_indeksler, min(len(basarisiz_indeksler), len(basarili_indeksler)))
            dengeli_indeksler = basarili_indeksler + secilen_basarisizlar
            random.shuffle(dengeli_indeksler)  # Karıştır
            
            X_train = [X_train_raw[i] for i in dengeli_indeksler]
            Y_train = [Y_train_raw[i] for i in dengeli_indeksler]
        else:
            X_train, Y_train = X_train_raw, Y_train_raw
    else:
        # Tüm örnekleri kullan, sadece karıştır
        indeksler = list(range(len(Y_train_raw)))
        random.shuffle(indeksler)
        X_train = [X_train_raw[i] for i in indeksler]
        Y_train = [Y_train_raw[i] for i in indeksler]
        
    print(f"Train Dengeleme Sonrası -> Kullanılacak Örnek Sayısı: {len(X_train)}")
    print(f"Validasyon Seti (Doğal Akışta) -> Örnek Sayısı: {len(X_val_raw)}")
    
    # PyTorch Tensörlerine Çevir (Torch yüklü değilse NumPy fallback)
    if TORCH_AVAILABLE:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
        
        if len(X_val_raw) > 0:
            X_val_tensor = torch.tensor(X_val_raw, dtype=torch.float32)
            Y_val_tensor = torch.tensor(Y_val_raw, dtype=torch.float32).unsqueeze(1)
        else:
            X_val_tensor = torch.empty((0, GECMIS_MUM_SAYISI, len(ozellikler)), dtype=torch.float32)
            Y_val_tensor = torch.empty((0, 1), dtype=torch.float32)
            
        return X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor, scaler
    else:
        return np.array(X_train), np.array(Y_train).reshape(-1, 1), np.array(X_val_raw), np.array(Y_val_raw).reshape(-1, 1), scaler

def tensore_donustur_hiyerarsik(df_or_list, islem_yonu="LONG", val_oran=0.15):
    """
    5 Katmanlı özellikleri Makro ve Teknik/Yapısal olarak iki ayrı kola böler.
    Veri sızıntısını (Data Leakage) önlemek için Scaler SADECE eğitim (Train)
    verisinde fit edilir. Sınıf dengesizliği sadece Train setinde çözülür.
    """
    import random
    print(f"\n[{islem_yonu} BEYNİ] Veriler Çok Kollu Yapay Zeka Tensörlerine dönüştürülüyor (Sızıntı Korumalı)...")
    
    # 1. KATMAN: MAKROEKONOMİK VE SEKTÖREL VERİLER (Hava Durumu ve Akıllı Para)
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
    
    if isinstance(df_or_list, list):
        dfs_list = df_or_list
    elif isinstance(df_or_list, tuple):
        dfs_list = list(df_or_list)
    else:
        dfs_list = [df_or_list]
        
    scaler_makro = CustomMinMaxScaler(feature_range=(-1, 1))
    scaler_teknik = CustomMinMaxScaler(feature_range=(-1, 1))
    
    # ==========================================
    # 1. ADIM: SADECE EĞİTİM VERİSİYLE SCALER'I EĞİT (Veri Sızıntısını Önleme)
    # ==========================================
    train_makro_list = []
    train_teknik_list = []
    
    for df in dfs_list:
        if df.empty or len(df) < 100: continue
        # Geleceği görmemek için sadece kronolojik olarak ilk %85'i alıyoruz
        split_idx = int(len(df) * (1.0 - val_oran))
        train_makro_list.append(df[makro_ozellikler].iloc[:split_idx].values)
        train_teknik_list.append(df[teknik_ozellikler].iloc[:split_idx].values)
        
    if not train_makro_list:
        raise ValueError("HATA: Boş veri listesi veya yetersiz veri!")
        
    scaler_makro.fit_transform(np.concatenate(train_makro_list, axis=0))
    scaler_teknik.fit_transform(np.concatenate(train_teknik_list, axis=0))
    
    # ==========================================
    # 2. ADIM: PENCERELERİ (SEQUENCES) OLUŞTUR VE TRAIN/VAL OLARAK BÖL
    # ==========================================
    X_makro_train, X_teknik_train, Y_train = [], [], []
    X_makro_val, X_teknik_val, Y_val = [], [], []
    
    GECMIS_MUM_SAYISI = 60
    
    for df in dfs_list:
        if df.empty or len(df) < GECMIS_MUM_SAYISI + 1: continue
            
        # Artık tüm veriyi scale ederken, sadece geçmişi öğrenmiş scaler'ı kullanıyoruz
        X_makro_olcekli = scaler_makro.transform(df[makro_ozellikler].values)
        X_teknik_olcekli = scaler_teknik.transform(df[teknik_ozellikler].values)
        Y_ham = df['Hedef_Yonu_Long'].values if islem_yonu == "LONG" else df['Hedef_Yonu_Short'].values
        
        # O hisse için tüm kayan pencereleri oluştur (Meta-Labeling Sinyal Filtresi)
        makro_seq, teknik_seq, y_seq = [], [], []
        
        for i in range(len(df) - GECMIS_MUM_SAYISI):
            # Hedef anındaki satırı al (Sinyalin tetiklendiği an)
            hedef_satir = df.iloc[i + GECMIS_MUM_SAYISI - 1]
            
            # SADECE MİKRO-YAPI SİNYALİ YANAN GÜNLERİ EĞİTİM SETİNE AL (Meta-Labeling)
            if islem_yonu == "LONG":
                sinyal_var_mi = (hedef_satir['Boga_Uyumsuzlugu'] == 1) or \
                                (hedef_satir['Alt_Golge_Orani'] >= 0.70) or \
                                (hedef_satir['Alt_Fitil_Gucu'] > 2.0)
            else: # SHORT beynini eğitiyorsak
                sinyal_var_mi = (hedef_satir['Ayi_Uyumsuzlugu'] == 1) or \
                                (hedef_satir['Ust_Golge_Orani'] >= 0.70) or \
                                (hedef_satir['Ust_Fitil_Gucu'] > 2.0)
                                
            # Eğer o gün piyasada temel stratejimizin bir sinyali yoksa, o günü es geç!
            if not sinyal_var_mi:
                continue

            teknik_seq.append(X_teknik_olcekli[i : (i + GECMIS_MUM_SAYISI)])
            makro_seq.append(X_makro_olcekli[i + GECMIS_MUM_SAYISI - 1])
            y_seq.append(Y_ham[i + GECMIS_MUM_SAYISI])
            
        if not y_seq: continue
        
        # Oluşan pencereleri kronolojik olarak Train ve Val'a dağıt
        split_seq_idx = int(len(y_seq) * (1.0 - val_oran))
        
        # AFML SIZINTI KORUMASI: PURGING & EMBARGO
        # 1. PURGING: Eğitim setindeki son işlemin sonucu belli olana kadar geçecek zaman (ZAMAN_BARIYERI)
        # 2. EMBARGO: Modelin Validation setinin başını sezmemesi için ekstra izolasyon boşluğu
        ZAMAN_BARIYERI = 15 
        PURGE_GAP = ZAMAN_BARIYERI 
        EMBARGO_GAP = GECMIS_MUM_SAYISI 
        
        # Eğitim setini, Val setinin başladığı yerden çok daha geride bitirmeliyiz
        train_end_idx = split_seq_idx - (PURGE_GAP + EMBARGO_GAP)
        
        if train_end_idx <= 0:
            train_end_idx = len(y_seq)
            val_start_idx = len(y_seq)
        else:
            val_start_idx = split_seq_idx
        
        X_makro_train.extend(makro_seq[:train_end_idx])
        X_teknik_train.extend(teknik_seq[:train_end_idx])
        Y_train.extend(y_seq[:train_end_idx])
        
        if val_start_idx < len(y_seq):
            X_makro_val.extend(makro_seq[val_start_idx:])
            X_teknik_val.extend(teknik_seq[val_start_idx:])
            Y_val.extend(y_seq[val_start_idx:])
        
    # ==========================================
    # 3. ADIM: SADECE KARIŞTIR — UNDERSAMPLING YOK
    # pos_weight ile kayıp fonksiyonu dengesizliği zaten halleder.
    # Undersampling, eğitim verisini miniaturize ediyordu ve modeli
    # negatif tahmine kilitleyen temel sorundu.
    # ==========================================
    pos_sayisi = sum(1 for y in Y_train if y == 1)
    neg_sayisi = len(Y_train) - pos_sayisi

    print(f"[{islem_yonu}] Train (Ham) -> Pozitif (1): {pos_sayisi}, Negatif (0): {neg_sayisi}, Toplam: {len(Y_train)}")
    print(f"[{islem_yonu}] Validasyon (Doğal Dağılım) -> Toplam: {len(Y_val)}")

    # pos_weight = neg/pos → Kayıp fonksiyonuna verilecek sınıf ağırlığı
    # Model artık nadir pozitif sınıfı 'neg/pos' kat daha önemli görür
    if pos_sayisi > 0 and neg_sayisi > 0:
        pos_weight_ratio = neg_sayisi / pos_sayisi
    else:
        pos_weight_ratio = 1.0

    print(f"[{islem_yonu}] pos_weight oranı (neg/pos): {pos_weight_ratio:.2f}x")

    # Karıştır (zamansal blok değil, global shuffle — train içinde)
    indeksler = list(range(len(Y_train)))
    random.shuffle(indeksler)
    X_makro_train = [X_makro_train[i] for i in indeksler]
    X_teknik_train = [X_teknik_train[i] for i in indeksler]
    Y_train = [Y_train[i] for i in indeksler]

    print(f"[{islem_yonu}] Eğitim hazır -> {len(Y_train)} örnek (Undersampling YOK, pos_weight={pos_weight_ratio:.1f}x)")

    # PyTorch Tensörlerine Dönüştürme
    if TORCH_AVAILABLE:
        X_m_tr = torch.tensor(np.array(X_makro_train), dtype=torch.float32)
        X_t_tr = torch.tensor(np.array(X_teknik_train), dtype=torch.float32)
        Y_tr   = torch.tensor(np.array(Y_train), dtype=torch.float32).unsqueeze(1)

        X_m_v = torch.tensor(np.array(X_makro_val), dtype=torch.float32)
        X_t_v = torch.tensor(np.array(X_teknik_val), dtype=torch.float32)
        Y_v   = torch.tensor(np.array(Y_val), dtype=torch.float32).unsqueeze(1)

        return X_m_tr, X_t_tr, Y_tr, X_m_v, X_t_v, Y_v, scaler_makro, scaler_teknik, pos_weight_ratio
    else:
        return (np.array(X_makro_train), np.array(X_teknik_train), np.array(Y_train).reshape(-1,1),
                np.array(X_makro_val),   np.array(X_teknik_val),   np.array(Y_val).reshape(-1,1),
                scaler_makro, scaler_teknik, pos_weight_ratio)
