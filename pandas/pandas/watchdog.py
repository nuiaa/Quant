# -*- coding: utf-8 -*-
"""
AHTAPOT QUANT - ÇAPRAZ PLATFORM (WINDOWS & LINUX) 7/24 GÖZLEMCİ MOTORU
Bu motor, ana botu (proje2.py) korumalı bir fanus içinde çalıştırır.
Çökme durumunda işletim sisteminden bağımsız olarak botu diriltir.
"""

import subprocess
import time
import platform
import os
from datetime import datetime

# Çalıştırılacak ana bot dosyan
ANA_BOT_DOSYASI = "proje2.py"
# Çökme durumunda sistemi dinlendirme süresi (API banı yememek ve RAM'i boşaltmak için)
YENIDEN_BASLATMA_GECIKMESI = 15

def log_yaz(mesaj):
    """Gözlemci kayıtlarını hem ekrana hem log dosyasına yazar."""
    zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tam_mesaj = f"[{zaman}] [WATCHDOG] {mesaj}"
    print(tam_mesaj)
    with open("watchdog_otopsi.log", "a", encoding="utf-8") as f:
        f.write(tam_mesaj + "\n")

def isletim_sistemini_tanila():
    """Çalıştığı donanımı ve OS'i algılar."""
    os_adi = platform.system()
    mimari = platform.machine()
    log_yaz(f"Sistem Algılandı: {os_adi} ({mimari})")
    
    # Python komutunu OS'e göre belirle
    if os_adi == "Windows":
        return "python"
    else:
        # Linux dağıtımları için
        return "python3"

def otonom_donguyu_baslat():
    python_komutu = isletim_sistemini_tanila()
    
    while True:
        log_yaz(f"🚀 Ana motor ({ANA_BOT_DOSYASI}) başlatılıyor...")
        
        try:
            # Botu alt süreç olarak başlat (Bloklayıcı çağrı - bot kapanana kadar burada bekler)
            # stdout ve stderr doğrudan terminale akar
            surec = subprocess.Popen([python_komutu, ANA_BOT_DOSYASI])
            
            # Watchdog, botun çalışmasını bekliyor
            surec.wait()
            
            # Eğer kod buraya ulaştıysa, bot kapanmış demektir. Çıkış kodunu al.
            cikis_kodu = surec.returncode
            
            if cikis_kodu == 0:
                log_yaz("Bot normal ve güvenli bir şekilde kapatıldı. Yeniden başlatılmıyor.")
                break # Kasıtlı kapatıldıysa döngüyü kır
            else:
                log_yaz(f"⚠️ DİKKAT: Bot beklenmedik şekilde çöktü! (Çıkış Kodu: {cikis_kodu})")
                
        except Exception as e:
            log_yaz(f"❌ Kritik Sistem Hatası: {e}")
            
        log_yaz(f"🔄 Sistem {YENIDEN_BASLATMA_GECIKMESI} saniye içinde RAM temizlenip yeniden başlatılacak...")
        time.sleep(YENIDEN_BASLATMA_GECIKMESI)

if __name__ == "__main__":
    otonom_donguyu_baslat()
