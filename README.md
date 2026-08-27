# 📈 Quant AI Trading Bot & Finansal Analiz Sistemi
Bu proje, hisse senetleri ve piyasa verilerini analiz etmek, yapay zeka destekli alım-satım sinyalleri üretmek ve portföy yönetimini otomatize etmek için geliştirilmiş kapsamlı bir **Algoritmik Trading (Quant)** sistemidir.
Yapay zeka modelleri, SQLite tabanlı hızlı veri önbellekleme (caching), gelişmiş backtest motorları ve PyQt6 tabanlı kullanıcı dostu bir arayüz ile donatılmıştır.
## 🚀 Öne Çıkan Özellikler
* **🧠 Yapay Zeka ve Makine Öğrenmesi (`beyin_mimarisi.py`, `egitim_dongusu.py`)**: Piyasa verilerini işleyerek varlıkların yönünü tahmin eden derin öğrenme mimarisi.
* **📊 Gelişmiş Backtest Motoru**: 10 yıllık geriye dönük testler (`backtest_10y_gunluk.py`) ve kapsamlı strateji testleri (`kapsamli_backtest.py`).
* **🖥️ Modern Kullanıcı Arayüzü (`arayuz.py`)**: PyQt6 kullanılarak geliştirilmiş, portföy durumunu ve canlı analizleri görselleştiren şık masaüstü arayüzü.
* **⚡ Hızlı Veri İşleme ve Caching (`SQLiteCache`)**: Yahoo Finance (yfinance) isteklerini minimize etmek ve performansı artırmak için SQLite tabanlı yerel veri tabanı.
* **🌐 Makro ve Sektörel Analiz (`piyasa_haritasi.json`)**: Sektör, endüstri ve makro-ekonomik hassasiyetlere (VIX, DXY) göre dinamik "Varlık DNA" analizi.
* **📈 Opsiyon ve GEX Analizi (`opsiyon_gex_motoru.py`)**: Piyasa yapıcılık (Market Maker) seviyelerini ve Gamma Exposure (GEX) verilerini hesaba katan analiz modülü.
* **📱 Canlı/Sanal Portföy Yönetimi**: Gerçek (`canli_portfoy.json`) ve Sanal (`sanal_portfoy.json`) cüzdanlar üzerinden kar/zarar (PnL) takibi.
* **🔔 Telegram Entegrasyonu**: Alım/Satım sinyalleri ve kritik piyasa uyarıları için Telegram bot bildirimleri.
## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python 3.x
* **Arayüz:** PyQt6, PyQtGraph
* **Veri Analizi & AI:** Pandas, NumPy, PyTorch, yfinance
* **Veritabanı:** SQLite3
## 📂 Proje Yapısı
| Dosya / Klasör | Açıklama |
| :--- | :--- |
| proje2.py | Ana analiz, veri çekme ve botun karar mekanizması. |
| arayuz.py | PyQt6 tabanlı grafiksel kullanıcı arayüzü. |
| beyin_mimarisi.py | Yapay zeka modelinin (Neural Network/Deep Learning) mimarisi. |
| egitim_dongusu.py | Yapay zeka modelinin piyasa verileriyle eğitildiği döngü. |
| opsiyon_gex_motoru.py | Opsiyon zincirleri ve Gamma analizi modülü. |
| telemetri_motoru.py | Sistemin performansını ve anlık durumunu loglayan modül. |
| canli_portfoy.json / sanal_portfoy.json | Portföy durumunu tutan konfigürasyon dosyaları. |

## ⚙️ Kurulum ve Kullanım
1. **Repoyu Klonlayın:**
bash
git clone [https://github.com/nuiaa/pandas/pandas](https://github.com/nuiaa/Quant/tree/main)

Gerekli Kütüphaneleri Yükleyin:

pip install -r requirements.txt

Konfigürasyon (config.py): Projenin ana dizininde bir config.py dosyası oluşturun ve içerisine Telegram Bot bilgilerinizi (veya API anahtarlarınızı) girin:

TELEGRAM_BOT_TOKEN = "sizin_bot_tokeniniz"
TELEGRAM_CHAT_ID = "sizin_chat_id"

Projeyi Başlatın: Arayüzü başlatmak için:

python arayuz.py

 Yasal Uyarı
Bu proje yalnızca eğitim, araştırma ve test amaçlıdır. Bu yazılım tarafından üretilen hiçbir sinyal, veri veya analiz finansal bir tavsiye niteliği taşımaz. Gerçek parayla işlem yapmadan önce tüm riskleri kendi inisiyatifinizde değerlendirmelisiniz. Yazılımın kullanımından doğabilecek maddi kayıplardan geliştirici sorumlu tutulamaz.
