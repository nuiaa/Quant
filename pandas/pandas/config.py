# ==========================================
# TELEGRAM BİLDİRİM YAPILANDIRMASI (GÜVENLİ AŞAMA)
# ==========================================
# Kimlik bilgileri artık .env dosyasından okunur.
# .env dosyasını paylaşmayın veya sürüm kontrol sistemlerine (örn. git) eklemeyin.

import os

def _env_oku(anahtar, varsayilan=None):
    """Önce os.environ'dan, yoksa .env dosyasından okur."""
    deger = os.environ.get(anahtar)
    if deger:
        return deger
    # .env dosyasından oku (dotenv kütüphanesi gerektirmeden)
    env_dosyasi = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_dosyasi):
        try:
            with open(env_dosyasi, 'r', encoding='utf-8') as f:
                for satir in f:
                    satir = satir.strip()
                    if satir and not satir.startswith('#') and '=' in satir:
                        k, v = satir.split('=', 1)
                        if k.strip() == anahtar:
                            return v.strip()
        except Exception:
            pass
    return varsayilan

TELEGRAM_BOT_TOKEN = _env_oku("TELEGRAM_BOT_TOKEN", "BOT_TOKENINIZI_BURAYA_YAZIN")
TELEGRAM_CHAT_ID = _env_oku("TELEGRAM_CHAT_ID", "CHAT_IDNIZI_BURAYA_YAZIN")

# ==========================================
# PORTFÖY VE RİSK YÖNETİMİ YAPILANDIRMASI
# ==========================================
PORTFOLIO_SIZE = float(_env_oku("PORTFOLIO_SIZE", "10000.0"))
RISK_PERCENT = float(_env_oku("RISK_PERCENT", "0.02"))
