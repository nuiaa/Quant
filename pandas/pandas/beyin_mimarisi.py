import numpy as np

# PyTorch kütüphanesini dinamik olarak içe aktar (import) ve kontrol et
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # PyTorch kurulu değilse kodun yapısının incelenmesi için sahte nn.Module sınıfı tanımlayalım
    class nn_mock:
        class Module:
            pass
    nn = nn_mock()

# ==========================================
# HIBRIT BEYIN MIMARISI (LSTM + ATTENTION)
# ==========================================
if TORCH_AVAILABLE:
    class HibritQuantModeli(nn.Module):
        def __init__(self, girdi_sayisi, gizli_katman_boyutu=64, lstm_katman_sayisi=2, dikkat_baslik_sayisi=4, dropout_orani=0.2):
            """
            KANTİTATİF YAPAY ZEKA BEYNİ
            - girdi_sayisi: Özellik (Feature) sayımız (Örn: Fiyat farkları, RSI, MACD vs. toplam 8 veya 13)
            - gizli_katman_boyutu: Beynin düşünme kapasitesi (Nöron sayısı)
            - lstm_katman_sayisi: Kaç katmanlı bir hafıza istiyoruz? (Kısa vadeli hafıza)
            - dikkat_baslik_sayisi: Transformer'ın veriye kaç farklı açıdan odaklanacağı.
            """
            super(HibritQuantModeli, self).__init__()
            
            self.gizli_katman_boyutu = gizli_katman_boyutu
            
            # 1. BÖLÜM: KISA VADELİ HAFIZA (LSTM)
            # Geçmiş 60 mumluk serüveni sırasıyla okur ve piyasanın zamana bağlı momentumunu kavrar.
            self.lstm = nn.LSTM(
                input_size=girdi_sayisi, 
                hidden_size=gizli_katman_boyutu, 
                num_layers=lstm_katman_sayisi, 
                batch_first=True, 
                dropout=dropout_orani if lstm_katman_sayisi > 1 else 0
            )
            
            # 2. BÖLÜM: BÜYÜK RESİM / ODAK (Transformer - Multi-Head Self Attention)
            # HFT fonlarının sırrı: LSTM'in ürettiği tüm zaman adımlarına bakar,
            # gürültüyü (yatay piyasayı) filtreler ve ani hacim/fiyat patlamalarına (kritik mumlara) odaklanır.
            self.dikkat_mekanizmasi = nn.MultiheadAttention(
                embed_dim=gizli_katman_boyutu, 
                num_heads=dikkat_baslik_sayisi, 
                batch_first=True
            )
            
            # 3. BÖLÜM: KARAR MEKANİZMASI (Tam Bağlı Katmanlar - FNN)
            # Elde edilen süzülmüş odak bilgisini Al/Sat sinyali olasılığına dönüştürür.
            self.fc1 = nn.Linear(gizli_katman_boyutu, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_orani)
            self.fc2 = nn.Linear(32, 1)
            
            # Çıktıyı 0 ile 1 arasında bir "Olasılık" (% İhtimal) değerine sıkıştırır.
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x'in Şekli: (Batch, 60 Mum, Özellik Sayısı)
            
            # 1. Adım: Veriyi hafıza ağından geçir (LSTM)
            lstm_ciktisi, (h_n, c_n) = self.lstm(x)
            # lstm_ciktisi şekli: (Batch, 60 Mum, Gizli Katman Boyutu)
            
            # 2. Adım: Model kendi kendine en önemli anlara odaklanıyor (Self-Attention)
            dikkat_ciktisi, dikkat_agirliklari = self.dikkat_mekanizmasi(
                lstm_ciktisi, lstm_ciktisi, lstm_ciktisi
            )
            
            # Bize sadece en son anın (60. mumun) tüm geçmiş odaklı süzülmüş bağlamı (Context) lazım
            son_an_baglami = dikkat_ciktisi[:, -1, :]
            
            # 3. Adım: Karar katmanlarından geçir
            karar = self.fc1(son_an_baglami)
            karar = self.relu(karar)
            karar = self.dropout(karar)
            karar = self.fc2(karar)
            
            # 4. Adım: Yüzdelik olasılığa dönüştür (Örn: 0.78 -> %78 Yükseliş İhtimali)
            olasilik = self.sigmoid(karar)
            
            return olasilik
else:
    # PyTorch yüklü değilse kullanıcının kodu inceleyebilmesi ve simüle edebilmesi için bilgilendirici sınıf
    class HibritQuantModeli:
        def __init__(self, girdi_sayisi, gizli_katman_boyutu=64, lstm_katman_sayisi=2, dikkat_baslik_sayisi=4, dropout_orani=0.2):
            self.girdi_sayisi = girdi_sayisi
            self.gizli_katman_boyutu = gizli_katman_boyutu
            print("[Mock Model] HibritQuantModeli nesnesi basariyla olusturuldu (NumPy Simulasyon Modu).")

        def __call__(self, x):
            # Sahte tahmin üret (0 ile 1 arasında sigmoid benzeri)
            batch_boyutu = x.shape[0]
            tahminler = np.random.uniform(0.1, 0.9, size=(batch_boyutu, 1))
            return tahminler

# ==========================================
# DINAMIK HIYERARSIK AHTAPOT BEYIN MIMARISI
# ==========================================
if TORCH_AVAILABLE:
    class DinamikHiyerarsikModel(nn.Module):
        def __init__(self, makro_girdi_sayisi=13, teknik_girdi_sayisi=30, gizli_katman_boyutu=64, lstm_katman_sayisi=2, dikkat_baslik_sayisi=4, dropout_orani=0.4):
            """
            Hiyerarşik Çift Kollu Ahtapot (Fusion) Ağ Mimarisi
            - makro_girdi_sayisi: Makroekonomik anlık özellikler (VIX, DXY + 7 Sektör + 4 Zaman -> 13 özellik)
            - teknik_girdi_sayisi: Teknik ve yapısal zaman serisi özellikleri (20 özellik)
            """
            super(DinamikHiyerarsikModel, self).__init__()
            
            # 1. KOL: TEKNİK ZAMAN SERİSİ HAFIZA AĞI (LSTM + Attention)
            self.lstm = nn.LSTM(
                input_size=teknik_girdi_sayisi, 
                hidden_size=gizli_katman_boyutu, 
                num_layers=lstm_katman_sayisi, 
                batch_first=True, 
                dropout=dropout_orani if lstm_katman_sayisi > 1 else 0
            )
            
            self.dikkat_mekanizmasi = nn.MultiheadAttention(
                embed_dim=gizli_katman_boyutu, 
                num_heads=dikkat_baslik_sayisi, 
                batch_first=True
            )
            
            # 2. KOL: MAKRO BİLGİ PROJEKSİYONU (Dense MLP)
            self.makro_fc = nn.Sequential(
                nn.Linear(makro_girdi_sayisi, 16),
                nn.ReLU(),
                nn.Dropout(dropout_orani)
            )
            
            self.fc1 = nn.Linear(gizli_katman_boyutu + 16, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_orani)
            self.fc2 = nn.Linear(32, 1)
            # Sigmoid kaldırıldı — BCEWithLogitsLoss eğitimde içeride sigmoid uygular

        def forward(self, makro_x, teknik_x):
            # 1. Kol: Teknik zaman serisi (LSTM + Attention)
            # teknik_x şekli: (Batch, 60, 15)
            lstm_ciktisi, _ = self.lstm(teknik_x) # (Batch, 60, 64)
            
            dikkat_ciktisi, _ = self.dikkat_mekanizmasi(
                lstm_ciktisi, lstm_ciktisi, lstm_ciktisi
            ) # (Batch, 60, 64)
            
            # En son anın süzülmüş odak bilgisini alıyoruz
            son_an_baglami = dikkat_ciktisi[:, -1, :] # (Batch, 64)
            
            # 2. Kol: Makro anlık hava durumu
            # makro_x şekli: (Batch, 2)
            makro_projeksiyon = self.makro_fc(makro_x) # (Batch, 16)
            
            # 3. İki kolu birleştir (Concatenate)
            fusion = torch.cat((son_an_baglami, makro_projeksiyon), dim=1) # (Batch, 80)
            
            # 4. Karar katmanları
            karar = self.fc1(fusion)
            karar = self.relu(karar)
            karar = self.dropout(karar)
            karar = self.fc2(karar)

            # Ham logit döndür (Sigmoid YOK — BCEWithLogitsLoss eğitimde, torch.sigmoid inference'da)
            return karar
else:
    # PyTorch yoksa NumPy simülasyonu
    class DinamikHiyerarsikModel:
        def __init__(self, makro_girdi_sayisi=13, teknik_girdi_sayisi=30, gizli_katman_boyutu=64, lstm_katman_sayisi=2, dikkat_baslik_sayisi=4, dropout_orani=0.4):
            self.makro_girdi_sayisi = makro_girdi_sayisi
            self.teknik_girdi_sayisi = teknik_girdi_sayisi
            print("[Mock Model] DinamikHiyerarsikModel nesnesi oluşturuldu (NumPy Simülasyon Modu).")

        def __call__(self, makro_x, teknik_x):
            batch_boyutu = teknik_x.shape[0]
            tahminler = np.random.uniform(0.1, 0.9, size=(batch_boyutu, 1))
            return tahminler

# ==========================================================
# ENTEGRASYON: PANDAS DATAFRAME'DEN 3D TENSÖR YAPILANDIRMA
# ==========================================
def veriyi_3d_tensore_donustur(df, ozellik_sutunlari, dizi_boyutu=60, device="cpu"):
    """
    Soğuk depodan okunup RAM'de işlenen 2D Pandas DataFrame'i,
    LSTM ve Transformer modelinin beklediği 3 boyutlu rolling sequence (Batch, Dizi_Boyutu, Özellikler)
    yapısına ve PyTorch tensörüne dönüştürür.
    
    Parametreler:
    - df: Pandas DataFrame (Özellikleri hesaplanmış sıcak işlem verisi)
    - ozellik_sutunlari: Modelin eğitileceği sütunların listesi (Örn: 8 indikatör)
    - dizi_boyutu: Geçmiş hafıza pencere büyüklüğü (Film şeridi uzunluğu, Varsayılan: 60)
    - device: Hedef cihaz ('cpu' veya 'cuda' / GPU)
    """
    # Özellik verilerini çek
    ozellikler_veri = df[ozellik_sutunlari].values
    N = len(ozellikler_veri)
    
    if N < dizi_boyutu:
        raise ValueError(f"Hata: Veri satır sayısı ({N}), geçmiş pencere boyutundan ({dizi_boyutu}) küçük olamaz!")
    
    # Kayan pencereler oluştur (Rolling Windows)
    # x_pencereler[i] = [i'den i+60'a kadar olan 60 mumluk matris]
    x_listesi = []
    for i in range(N - dizi_boyutu + 1):
        x_listesi.append(ozellikler_veri[i : i + dizi_boyutu])
        
    x_3d_np = np.array(x_listesi) # Şekil: (N - 60 + 1, 60, Özellik_Sayısı)
    
    # PyTorch yüklü ise tensöre çevir ve GPU/CPU'ya fırlat
    if TORCH_AVAILABLE:
        x_tensor = torch.tensor(x_3d_np, dtype=torch.float32)
        if device == "cuda" and torch.cuda.is_available():
            x_tensor = x_tensor.to("cuda")
        return x_tensor
    else:
        # PyTorch yoksa NumPy array olarak döndür
        return x_3d_np


# ==========================================
# TEST: BEYNİ AKTİF ETME VE SİMÜLASYON
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("HIBRIT BEYIN MODELI (LSTM + MULTI-HEAD ATTENTION) TEST PANELİ")
    print("=" * 60)
    
    girdi_ozellik_sayisi = 15
    gecmis_pencere = 60
    batch_boyutu = 128
    
    if TORCH_AVAILABLE:
        print("[Sistem Bilgisi] PyTorch aktif! Gercek derin ogrenme tensorleri olusturuluyor...")
        
        # Test icin sahte bir 3D tensor olusturalim: 
        # (128 adet veri ornegi, her biri 60 mumluk tarihce, 8 farkli indikator/ozellik)
        ornek_tensor = torch.rand((batch_boyutu, gecmis_pencere, girdi_ozellik_sayisi)) 
        
        # Hibrit Quant Modelini ayaga kaldir
        model = HibritQuantModeli(girdi_sayisi=girdi_ozellik_sayisi)
        
        # Tensoru modelin icine yolla
        tahminler = model(ornek_tensor)
        
        print("\nMODEL BASARIYLA CALISTIRILDI!")
        print(f"  * Girdi Sekli (Batch, Sequence, Features) : {list(ornek_tensor.shape)}")
        print(f"  * Model Karar Cikti Sekli (Sigmoid)      : {list(tahminler.shape)}")
        print("\nIlk 5 Karar Olasiligi (% Yukselis Ihtimali):")
        kararlar = tahminler[:5].detach().cpu().numpy()
        for idx, k in enumerate(kararlar):
            print(f"    Ornek {idx+1}: %{k[0]*100:.2f} ihtimalle Yukselis")
            
    else:
        print("[Sistem Bilgisi] Sisteminizde PyTorch (torch) bulunamadi.")
        print("Lutfen terminalinizde 'pip install torch' calistirarak derin ogrenmeyi aktiflestirin.")
        print("\n--- NUMPY SIMULASYON MODUNDA CALISTIRILIYOR ---")
        
        # NumPy ile sahte 3D matris olustur
        ornek_matris = np.random.rand(batch_boyutu, gecmis_pencere, girdi_ozellik_sayisi)
        
        model = HibritQuantModeli(girdi_sayisi=girdi_ozellik_sayisi)
        tahminler = model(ornek_matris)
        
        print(f"  * Girdi Sekli (NumPy 3D)                  : {ornek_matris.shape}")
        print(f"  * Simule Cikti Sekli (NumPy 2D)           : {tahminler.shape}")
        print("\nSimule Edilen Ilk 5 Yukselis Olasiligi:")
        for idx, k in enumerate(tahminler[:5]):
            print(f"    Ornek {idx+1}: %{k[0]*100:.2f} ihtimalle Yukselis")
            
    print("=" * 60)
