import sys
import json
import os
import subprocess
import csv
import random
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import yfinance as yf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QFrame, QScrollArea, QSizePolicy, QGridLayout,
    QStackedWidget
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QFont, QColor, QPainter

import pyqtgraph as pg

# ─────────────────────────────────────────────────────────────
# DOSYA YOLLARI
# ─────────────────────────────────────────────────────────────
CANLI_PORTFOY  = "canli_portfoy.json"
CANLI_HESAP    = "canli_hesap.json"
CANLI_GECMIS   = "canli_gecmis.csv"
SANAL_PORTFOY  = "sanal_portfoy.json"
SANAL_HESAP    = "sanal_hesap.json"
SANAL_GECMIS   = "sanal_gecmis.csv"

# ─────────────────────────────────────────────────────────────
# RENK PALETİ
# ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0a0c0f",
    "bg2":      "#0f1215",
    "bg3":      "#141820",
    "bg4":      "#1a1f2a",
    "bg5":      "#1e2535",
    "border":   "#1e2535",
    "border2":  "#252d3d",
    "accent":   "#00d4aa",
    "blue":     "#0088ff",
    "red":      "#ff4d6d",
    "amber":    "#f5a623",
    "purple":   "#a855f7",
    "text1":    "#e8edf5",
    "text2":    "#8a94a8",
    "text3":    "#4a5568",
}

MONO  = "JetBrains Mono"
SANS  = "Syne"

# ─────────────────────────────────────────────────────────────
# ARKA PLAN MOTORU
# ─────────────────────────────────────────────────────────────
class BotMotoru(QThread):
    log_sinyali   = pyqtSignal(str)
    bitis_sinyali = pyqtSignal()

    def __init__(self, komut_listesi):
        super().__init__()
        self.komut_listesi = komut_listesi
        self.surec         = None
        self.calisiyor     = True

    def run(self):
        try:
            self.surec = subprocess.Popen(
                self.komut_listesi,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8'
            )
            for satir in iter(self.surec.stdout.readline, ''):
                if not self.calisiyor:
                    break
                if satir:
                    self.log_sinyali.emit(satir.strip())
            self.surec.stdout.close()
            self.surec.wait()
        except Exception as e:
            self.log_sinyali.emit(f"[KRİTİK HATA] Motor Başlatılamadı: {e}")
        finally:
            self.bitis_sinyali.emit()

    def durdur(self):
        self.calisiyor = False
        if self.surec:
            try:
                self.surec.kill()  # terminate() yerine kill() — alt süreçleri de temizler
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────
# YARDIMCI: ÇERÇEVE
# ─────────────────────────────────────────────────────────────
def card(parent=None, bg=None, border=None, radius=10, padding="0px"):
    bg_color     = bg     or C["bg3"]
    border_color = border or C["border"]
    f = QFrame(parent)
    f.setStyleSheet(f"""
        QFrame {{
            background: {bg_color};
            border: 1px solid {border_color};
            border-radius: {radius}px;
        }}
    """)
    return f

def label(text="", color=None, size=11, bold=False, mono=False, parent=None):
    lbl  = QLabel(text, parent)
    font = QFont(MONO if mono else SANS, size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color or C['text1']}; background: transparent; border: none;")
    return lbl

def section_title(text):
    lbl = label(text.upper(), color=C["text3"], size=9, mono=True)
    lbl.setStyleSheet(f"color:{C['text3']}; background:transparent; border:none; letter-spacing: 1px;")
    return lbl

def tag_label(text, is_long=True):
    color  = C["accent"] if is_long else C["red"]
    bg     = "#0d2e25" if is_long else "#2e0d15"
    border = "#1a5c45" if is_long else "#5c1a27"
    lbl = QLabel(text)
    lbl.setFont(QFont(MONO, 9, QFont.Weight.Bold))
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFixedSize(52, 20)
    lbl.setStyleSheet(f"""
        QLabel {{
            color: {color};
            background: {bg};
            border: 1px solid {border};
            border-radius: 4px;
        }}
    """)
    return lbl

def divider(horizontal=True, parent=None):
    line = QFrame(parent)
    line.setFrameShape(QFrame.Shape.HLine if horizontal else QFrame.Shape.VLine)
    line.setStyleSheet(f"color: {C['border']}; background: {C['border']}; border: none; max-height: 1px;")
    return line

# ─────────────────────────────────────────────────────────────
# P&L GRAFİK WİDGETI (pyqtgraph)
# ─────────────────────────────────────────────────────────────
class PnLGrafik(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(C["bg3"])
        self.getPlotItem().hideAxis('top')
        self.getPlotItem().hideAxis('right')
        self.showGrid(x=True, y=True, alpha=0.07)

        ax = self.getPlotItem().getAxis('left')
        ay = self.getPlotItem().getAxis('bottom')
        ax.setTextPen(pg.mkPen(C["text3"]))
        ay.setTextPen(pg.mkPen(C["text3"]))
        ax.setPen(pg.mkPen(C["border"]))
        ay.setPen(pg.mkPen(C["border"]))

        self.getPlotItem().setContentsMargins(8, 8, 8, 8)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.x_data     = list(range(1, 61))
        self.y_canli    = [0.0] * 60
        self.y_sanal    = [0.0] * 60
        self.is_seeded  = False
        self.adim       = 60

        # Renkli dolgu için FillBetweenItem
        self.canli_egri = self.plot(self.x_data, self.y_canli,
            pen=pg.mkPen(C["accent"], width=2), name="Canlı P&L")
        self.sanal_egri = self.plot(self.x_data, self.y_sanal,
            pen=pg.mkPen(C["purple"], width=2, style=Qt.PenStyle.DashLine), name="Sanal P&L")

    def guncelle(self, canli_pnl, sanal_pnl):
        if not self.is_seeded:
            self.y_canli = [canli_pnl] * 60
            self.y_sanal = [sanal_pnl] * 60
            self.is_seeded = True
            
        self.adim += 1
        self.x_data.append(self.adim)
        self.y_canli.append(canli_pnl)
        self.y_sanal.append(sanal_pnl)
        if len(self.x_data) > 120:
            self.x_data.pop(0)
            self.y_canli.pop(0)
            self.y_sanal.pop(0)
        self.canli_egri.setData(self.x_data, self.y_canli)
        self.sanal_egri.setData(self.x_data, self.y_sanal)

# ─────────────────────────────────────────────────────────────
# CANDLESTICK GRAFİK WİDGETI (pyqtgraph)
# ─────────────────────────────────────────────────────────────
class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        super().__init__()
        self.data = data  # data must be a list of tuples (t, open, close, min, max)
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))
        w = (self.data[1][0] - self.data[0][0]) / 3. if len(self.data) > 1 else 0.3
        
        for (t, open, close, min, max) in self.data:
            color = C["accent"] if close >= open else C["red"]
            p.setPen(pg.mkPen(color, width=1))
            p.drawLine(pg.QtCore.QPointF(t, min), pg.QtCore.QPointF(t, max))
            p.setBrush(pg.mkBrush(color))
            p.drawRect(pg.QtCore.QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())

# ─────────────────────────────────────────────────────────────
# KPI KARTI
# ─────────────────────────────────────────────────────────────
class KPIKart(QFrame):
    def __init__(self, title, value, sub, accent_color, parent=None):
        super().__init__(parent)
        self.accent = accent_color
        self.setFixedHeight(90)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['bg3']};
                border: 1px solid {C['border']};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)

        self.lbl_title = section_title(title)
        self.lbl_val   = label(value, color=C["text1"], size=20, bold=True, mono=True)
        self.lbl_sub   = label(sub,   color=C["text3"], size=9,  mono=True)

        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_val)
        lay.addWidget(self.lbl_sub)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.accent))
        p.drawRoundedRect(0, 0, self.width(), 2, 1, 1)
        p.end()

    def set_value(self, val, sub=None, color=None):
        self.lbl_val.setText(val)
        if color:
            self.lbl_val.setStyleSheet(f"color:{color}; background:transparent; border:none;")
        if sub:
            self.lbl_sub.setText(sub)

# ─────────────────────────────────────────────────────────────
# LOG TERMİNALİ
# ─────────────────────────────────────────────────────────────
class LogTerminal(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont(MONO, 9))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {C['bg2']};
                color: {C['text1']};
                border: none;
                border-radius: 0px;
                padding: 8px;
            }}
            QScrollBar:vertical {{
                background: {C['bg2']};
                width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {C['border2']};
                border-radius: 3px;
            }}
        """)

    def log(self, mesaj, seviye="ok"):
        zaman = datetime.now().strftime("%H:%M:%S")
        renkler = {
            "ok":   C["accent"],
            "err":  C["red"],
            "warn": C["amber"],
            "info": C["blue"],
            "sys":  C["text2"],
        }
        r = renkler.get(seviye, C["text2"])

        # Seviye otomatik tespiti
        m = mesaj.lower()
        if any(k in m for k in ["hata", "error", "kritik", "stop loss"]):
            r = C["red"]
        elif any(k in m for k in ["uyarı", "warning", "ısınma"]):
            r = C["amber"]
        elif any(k in m for k in ["başlat", "sistem", "bilgi", "makro"]):
            r = C["blue"]
        elif any(k in m for k in ["✅", "tp hit", "kâr", "long", "eğitildi"]):
            r = C["accent"]

        from PyQt6.QtGui import QTextCursor
        
        bar = self.verticalScrollBar()
        is_at_bottom = bar.value() == bar.maximum()

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        html_msg = (
            f'<span style="color:{C["text3"]}">{zaman} </span>'
            f'<span style="color:{r}">{mesaj}</span>'
        )
        
        if not self.document().isEmpty():
            cursor.insertHtml("<br>" + html_msg)
        else:
            cursor.insertHtml(html_msg)

        if is_at_bottom:
            bar.setValue(bar.maximum())

# ─────────────────────────────────────────────────────────────
# PORTFÖY / GEÇMİŞ TABLO
# ─────────────────────────────────────────────────────────────
def build_table(kolonlar, parent=None):
    t = QTableWidget(0, len(kolonlar), parent)
    t.setHorizontalHeaderLabels(kolonlar)
    t.verticalHeader().setVisible(False)
    t.setShowGrid(False)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.setStyleSheet(f"""
        QTableWidget {{
            background: transparent;
            color: {C['text1']};
            border: none;
            font-family: '{MONO}';
            font-size: 11px;
            gridline-color: transparent;
        }}
        QTableWidget::item {{
            padding: 8px 12px;
            border-bottom: 1px solid {C['border']};
            background: transparent;
        }}
        QTableWidget::item:selected {{
            background: {C['bg4']};
            color: {C['text1']};
        }}
        QTableWidget::item:hover {{
            background: {C['bg4']};
        }}
        QHeaderView::section {{
            background: {C['bg3']};
            color: {C['text3']};
            font-family: '{MONO}';
            font-size: 9px;
            padding: 6px 12px;
            border: none;
            border-bottom: 1px solid {C['border']};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        QScrollBar:vertical {{
            background: {C['bg2']};
            width: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {C['border2']};
            border-radius: 2px;
        }}
    """)
    return t

# ─────────────────────────────────────────────────────────────
# SEKTÖR KART
# ─────────────────────────────────────────────────────────────
class SektorKart(QFrame):
    def __init__(self, isim, pct, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['bg3']};
                border: 1px solid {C['border']};
                border-radius: 6px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(3)

        lbl_isim = QLabel(isim.upper())
        lbl_isim.setFont(QFont(MONO, 8))
        lbl_isim.setStyleSheet(f"color:{C['text3']}; background:transparent; border:none;")

        color = C["accent"] if pct >= 0 else C["red"]
        sign  = "+" if pct >= 0 else ""
        self.lbl_pct = QLabel(f"{sign}{pct:.2f}%")
        self.lbl_pct.setFont(QFont(MONO, 12, QFont.Weight.Bold))
        self.lbl_pct.setStyleSheet(f"color:{color}; background:transparent; border:none;")

        lay.addWidget(lbl_isim)
        lay.addWidget(self.lbl_pct)

    def guncelle(self, pct):
        color = C["accent"] if pct >= 0 else C["red"]
        sign  = "+" if pct >= 0 else ""
        self.lbl_pct.setText(f"{sign}{pct:.2f}%")
        self.lbl_pct.setStyleSheet(f"color:{color}; background:transparent; border:none;")

# ─────────────────────────────────────────────────────────────
# BEYİN İLERLEME SATIRI
# ─────────────────────────────────────────────────────────────
class BeynilerlemeRow(QFrame):
    def __init__(self, isim, epoch, toplam, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lbl = QLabel(isim)
        lbl.setFont(QFont(MONO, 9))
        lbl.setFixedWidth(80)
        lbl.setStyleSheet(f"color:{C['text2']}; background:transparent; border:none;")

        self.bar_frame = QFrame()
        self.bar_frame.setFixedHeight(5)
        self.bar_frame.setStyleSheet(f"""
            QFrame {{ background: {C['bg4']}; border-radius: 2px; border: none; }}
        """)
        self.bar_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.bar_fill = QFrame(self.bar_frame)
        self.bar_fill.setFixedHeight(5)
        self.bar_fill.setStyleSheet(f"background: {C['accent']}; border-radius: 2px; border: none;")

        oran = min(epoch / toplam, 1.0)
        if epoch >= toplam:
            renk = C["accent"]
            etiket = "Hazır"
        else:
            renk = C["blue"] if oran > 0.5 else C["amber"]
            etiket = f"{epoch}/{toplam}"
            self.bar_fill.setStyleSheet(f"background: {renk}; border-radius: 2px; border: none;")

        self.lbl_epoch = QLabel(etiket)
        self.lbl_epoch.setFont(QFont(MONO, 9))
        self.lbl_epoch.setFixedWidth(52)
        self.lbl_epoch.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_epoch.setStyleSheet(f"color:{renk}; background:transparent; border:none;")

        lay.addWidget(lbl)
        lay.addWidget(self.bar_frame)
        lay.addWidget(self.lbl_epoch)

        # Bar genişliğini resizeEvent ile ayarla
        self._oran = oran

    def guncelle_durum(self, is_ready):
        if is_ready:
            self._oran = 1.0
            renk = C["accent"]
            etiket = "Hazır"
        else:
            self._oran = 0.0
            renk = C["red"]
            etiket = "Eksik"
            
        self.bar_fill.setStyleSheet(f"background: {renk}; border-radius: 2px; border: none;")
        self.lbl_epoch.setText(etiket)
        self.lbl_epoch.setStyleSheet(f"color:{renk}; background:transparent; border:none;")
        self.bar_fill.setFixedWidth(max(0, int(self.bar_frame.width() * self._oran)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.bar_frame.width()
        self.bar_fill.setFixedWidth(max(0, int(w * self._oran)))

# ─────────────────────────────────────────────────────────────
# ANA PENCERE
# ─────────────────────────────────────────────────────────────
class CommandCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantMaster · Kontrol Merkezi")
        self.setGeometry(60, 60, 1480, 860)
        self.setMinimumSize(1200, 720)
        self.setStyleSheet(f"background: {C['bg']}; color: {C['text1']};")

        self.aktif_motor  = None
        self.pnl_adim     = 0
        self.x_zaman      = []
        self.y_canli_pnl  = []
        self.y_sanal_pnl  = []
        
        self.gosterim_modu = "CANLI"

        merkez = QWidget()
        self.setCentralWidget(merkez)
        ana = QVBoxLayout(merkez)
        ana.setContentsMargins(0, 0, 0, 0)
        ana.setSpacing(0)

        # Topbar + İçerik
        ana.addWidget(self._topbar())
        icerik_widget = QWidget()
        icerik_lay = QHBoxLayout(icerik_widget)
        icerik_lay.setContentsMargins(0, 0, 0, 0)
        icerik_lay.setSpacing(0)

        self.stacked_widget = QStackedWidget()

        icerik_lay.addWidget(self._sidebar(), 0)
        icerik_lay.addWidget(self.stacked_widget, 1)
        
        self.page_dashboard = self._main_panel()
        self.page_portfoy = self._portfoy_page()
        self.page_gecmis = self._gecmis_page()
        self.page_canli = self._right_panel()
        self.page_sistem = self._sistem_page()
        self.page_qa = self._qa_page()
        self.page_grafik = self._canli_grafik_page()
        self.page_takip = self._takip_listesi_page()
        
        self.stacked_widget.addWidget(self.page_dashboard)  # 0
        self.stacked_widget.addWidget(self.page_portfoy)    # 1
        self.stacked_widget.addWidget(self.page_gecmis)     # 2
        self.stacked_widget.addWidget(self.page_canli)      # 3
        self.stacked_widget.addWidget(self.page_sistem)     # 4
        self.stacked_widget.addWidget(self.page_qa)         # 5
        self.stacked_widget.addWidget(self.page_grafik)     # 6
        self.stacked_widget.addWidget(self.page_takip)      # 7

        ana.addWidget(icerik_widget, 1)

        # Zamanlayıcı
        self.timer = QTimer()
        self.timer.timeout.connect(self.verileri_guncelle)
        self.timer.start(1000)
        self._makro_sayac = 0

    # ── TOPBAR ────────────────────────────────────────────────
    def _topbar(self):
        bar = QFrame()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"""
            QFrame {{
                background: {C['bg2']};
                border-bottom: 1px solid {C['border']};
                border-radius: 0px;
            }}
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(14)

        # Minimalist Logo removed for cleaner UI as requested
        # Pulse dot
        dot = QLabel("●")
        dot.setFont(QFont(SANS, 10))
        dot.setStyleSheet(f"color: {C['accent']}; background: transparent; border: none;")
        lay.addWidget(dot)

        self.lbl_status = label("CANLI MOD", color=C["text3"], size=9, mono=True)
        lay.addWidget(self.lbl_status)

        self.lbl_saat = label("", color=C["text3"], size=9, mono=True)
        lay.addWidget(self.lbl_saat)
        self._saat_guncelle()
        saat_timer = QTimer(self)
        saat_timer.timeout.connect(self._saat_guncelle)
        saat_timer.start(1000)

        # Dikey ayraç
        def sep():
            s = QFrame()
            s.setFrameShape(QFrame.Shape.VLine)
            s.setFixedWidth(1)
            s.setFixedHeight(24)
            s.setStyleSheet(f"background: {C['border']}; border: none;")
            return s

        lay.addStretch()
        lay.addWidget(sep())

        for baslik, attr in [("Toplam Bakiye", "lbl_top_bakiye"),
                              ("Net P&L",       "lbl_top_pnl"),
                              ("Bugün P&L",     "lbl_top_bugun")]:
            grp = QWidget()
            grp.setStyleSheet("background: transparent;")
            gl  = QVBoxLayout(grp)
            gl.setContentsMargins(8, 0, 8, 0)
            gl.setSpacing(1)
            v_lbl = label("$0.00", color=C["text1"], size=13, bold=True, mono=True)
            b_lbl = label(baslik,  color=C["text3"], size=8,  mono=True)
            b_lbl.setStyleSheet(f"color:{C['text3']}; background:transparent; border:none; letter-spacing:0.5px;")
            gl.addWidget(v_lbl)
            gl.addWidget(b_lbl)
            setattr(self, attr, v_lbl)
            lay.addWidget(grp)
            lay.addWidget(sep())

        return bar

    def _saat_guncelle(self):
        self.lbl_saat.setText(datetime.now().strftime("%H:%M:%S"))

    # ── SIDEBAR ───────────────────────────────────────────────
    def _sidebar(self):
        sb = QFrame()
        sb.setFixedWidth(210)
        sb.setStyleSheet(f"""
            QFrame {{
                background: {C['bg2']};
                border-right: 1px solid {C['border']};
                border-radius: 0px;
            }}
        """)
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(0, 12, 0, 12)
        lay.setSpacing(0)

        def nav_section(title):
            lbl = QLabel(title.upper())
            lbl.setFont(QFont(MONO, 8))
            lbl.setStyleSheet(f"color:{C['text3']}; background:transparent; padding: 14px 14px 6px 14px; letter-spacing:1px;")
            lay.addWidget(lbl)

        self.nav_buttons = []
        def nav_btn(icon, text, page_index):
            btn = QPushButton(f"  {icon}   {text}")
            btn.setFont(QFont(SANS, 11))
            btn.setProperty("page_index", page_index)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {C['text2']};
                    border: none;
                    border-radius: 6px;
                    padding: 9px 12px;
                    text-align: left;
                    margin: 1px 8px;
                }}
                QPushButton:hover {{
                    background: {C['bg4']};
                    color: {C['text1']};
                }}
            """)
            btn.clicked.connect(lambda _, idx=page_index: self._nav_clicked(idx))
            lay.addWidget(btn)
            self.nav_buttons.append(btn)
            return btn

        nav_section("Navigasyon")
        nav_btn("◈", "Dashboard",       0)
        nav_btn("◉", "Portföy",         1)
        nav_btn("≡", "İşlem Geçmişi",   2)
        nav_btn("📋", "Takip Listesi", 7)
        nav_btn("📊", "Canlı Grafik",   6)
        nav_btn("🔬", "Kalite Kontrol", 5)
        nav_btn("🌐", "Canlı Piyasa",    3)

        nav_section("Sistem")
        nav_btn("◈", "Sistem Paneli",   4)

        lay.addStretch()

        # Initialize first button as active
        self._nav_clicked(0)

        return sb

    def _nav_clicked(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for btn in self.nav_buttons:
            if btn.property("page_index") == index:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {C['bg4']};
                        color: {C['accent']};
                        border: none;
                        border-radius: 6px;
                        padding: 9px 12px;
                        text-align: left;
                        margin: 1px 8px;
                    }}
                    QPushButton:hover {{
                        background: {C['bg4']};
                        color: {C['text1']};
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent;
                        color: {C['text2']};
                        border: none;
                        border-radius: 6px;
                        padding: 9px 12px;
                        text-align: left;
                        margin: 1px 8px;
                    }}
                    QPushButton:hover {{
                        background: {C['bg4']};
                        color: {C['text1']};
                    }}
                """)

    # ── ANA PANEL ─────────────────────────────────────────────
    def _main_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {C['bg']}; border: none; }}
            QScrollBar:vertical {{ background: {C['bg']}; width: 5px; }}
            QScrollBar::handle:vertical {{ background: {C['border2']}; border-radius: 2px; }}
        """)

        container = QWidget()
        container.setStyleSheet(f"background: {C['bg']};")
        lay = QVBoxLayout(container)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(12)

        # KPI Satırı
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)
        self.kpi_pnl    = KPIKart("Net P&L",         "+$0",    "Başlangıçtan beri", C["accent"])
        self.kpi_wr     = KPIKart("Win Rate",         "0.0%",   "0 işlemden 0 kâr",  C["red"])
        self.kpi_pos    = KPIKart("Aktif Pozisyon",   "0",      "0 LONG · 0 SHORT",  C["blue"])
        self.kpi_sharpe = KPIKart("Sharpe Oranı",     "0.00",   "Son 30 gün",         C["amber"])
        for k in [self.kpi_pnl, self.kpi_wr, self.kpi_pos, self.kpi_sharpe]:
            kpi_row.addWidget(k)
        lay.addLayout(kpi_row)

        # P&L Grafik
        grafik_card = card()
        gc_lay = QVBoxLayout(grafik_card)
        gc_lay.setContentsMargins(0, 0, 0, 0)
        gc_lay.setSpacing(0)

        gh = QFrame()
        gh.setFixedHeight(42)
        gh.setStyleSheet(f"background: transparent; border-bottom: 1px solid {C['border']}; border-radius: 0px;")
        ghl = QHBoxLayout(gh)
        ghl.setContentsMargins(14, 0, 14, 0)
        g_title = section_title("P&L PERFORMANS GRAFİĞİ")
        ghl.addWidget(g_title)
        ghl.addSpacing(8)
        for txt, col, border in [("● CANLI", C["accent"], "#1a5c45"), ("● SANAL", C["purple"], "#3d2a7a")]:
            badge = QLabel(txt)
            badge.setFont(QFont(MONO, 8, QFont.Weight.Bold))
            badge.setStyleSheet(f"color:{col}; background:rgba(0,0,0,0.3); border:1px solid {border}; border-radius:8px; padding:2px 8px;")
            ghl.addWidget(badge)
        ghl.addStretch()

        gc_lay.addWidget(gh)
        self.pnl_grafik = PnLGrafik()
        gc_lay.addWidget(self.pnl_grafik)
        lay.addWidget(grafik_card)

        scroll.setWidget(container)
        return scroll

    def _portfoy_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.addWidget(self._portfoy_card())
        return w

    def _canli_grafik_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(10)
        
        from PyQt6.QtWidgets import QComboBox
        
        header = QHBoxLayout()
        title = section_title("CANLI FİYAT GRAFİĞİ")
        title.setFont(QFont(SANS, 14, QFont.Weight.Bold))
        header.addWidget(title)
        
        header.addStretch()
        
        lbl_sec = QLabel("Açık Pozisyon Seç:")
        lbl_sec.setStyleSheet(f"color: {C['text2']}; font-weight: bold;")
        header.addWidget(lbl_sec)
        
        self.combo_sembol = QComboBox()
        self.combo_sembol.setMinimumWidth(150)
        self.combo_sembol.setStyleSheet(f"""
            QComboBox {{
                background-color: {C['bg3']};
                color: {C['text1']};
                border: 1px solid {C['border']};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox::drop-down {{ border: none; }}
        """)
        self.combo_sembol.currentTextChanged.connect(self._on_combo_sembol_degisti)
        header.addWidget(self.combo_sembol)
        
        lay.addLayout(header)

        self.mum_grafik_widget = pg.PlotWidget()
        self.mum_grafik_widget.setBackground(C["bg3"])
        self.mum_grafik_widget.getPlotItem().hideAxis('top')
        self.mum_grafik_widget.getPlotItem().hideAxis('right')
        self.mum_grafik_widget.showGrid(x=True, y=True, alpha=0.07)
        ax = self.mum_grafik_widget.getPlotItem().getAxis('left')
        ay = self.mum_grafik_widget.getPlotItem().getAxis('bottom')
        ax.setTextPen(pg.mkPen(C["text3"]))
        ay.setTextPen(pg.mkPen(C["text3"]))
        ax.setPen(pg.mkPen(C["border"]))
        ay.setPen(pg.mkPen(C["border"]))
        
        lay.addWidget(self.mum_grafik_widget)
        
        return w

    def _gecmis_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.addWidget(self._gecmis_card())
        return w

    def _takip_listesi_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(12)
        
        lbl_baslik = label("GÜNCEL TAKİP LİSTESİ", size=18, bold=True, color=C["text1"], mono=True)
        lay.addWidget(lbl_baslik)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: transparent; }}")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0,0,0,0)
        scroll_lay.setSpacing(16)
        
        try:
            with open("aday_havuzu.json", "r", encoding="utf-8") as f:
                adaylar = json.load(f)
        except Exception:
            adaylar = {}
            
        try:
            with open("piyasa_haritasi.json", "r", encoding="utf-8") as f:
                ph = json.load(f)
            sektorler = ph.get("SEKTORLER", {})
        except Exception:
            sektorler = {}
            
        sym_to_sektor = {}
        for s_adi, s_veri in sektorler.items():
            for e_adi, e_veri in s_veri.get("Endustriler", {}).items():
                for sym in e_veri.get("Hisseler", []):
                    sym_to_sektor[sym] = s_adi
                    
        gruplar = {}
        for sym, d in adaylar.items():
            s = sym_to_sektor.get(sym, "Diğer")
            if s not in gruplar: gruplar[s] = []
            gruplar[s].append(d)
            
        for s_adi, items in sorted(gruplar.items()):
            from PyQt6.QtWidgets import QGroupBox
            grp = QGroupBox(f"{s_adi.upper()} ({len(items)} Hisse)")
            grp.setStyleSheet(f"QGroupBox {{ font-weight: bold; font-size: 14px; border: 1px solid {C['border']}; padding-top: 15px; color: {C['text1']}; }}")
            g_lay = QGridLayout(grp)
            
            row = 0
            col = 0
            for it in items:
                fiyat = it.get('Fiyat', 0)
                skor = it.get('Skor', 0)
                lbl = QLabel(f"• {it['Sembol']} (Skor: {skor} | Fiyat: ${fiyat:.2f})")
                lbl.setStyleSheet(f"color: {C['text2']}; font-size: 13px; font-family: '{MONO}';")
                g_lay.addWidget(lbl, row, col)
                col += 1
                if col > 3:
                    col = 0
                    row += 1
            scroll_lay.addWidget(grp)
            
        scroll_lay.addStretch()
        scroll.setWidget(scroll_content)
        lay.addWidget(scroll)
        
        return w

    def _qa_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(12)
        
        from PyQt6.QtWidgets import QGroupBox
        qa_grup = QGroupBox("Algoritma Performans Analizi")
        qa_grup.setStyleSheet(f"QGroupBox {{ font-weight: bold; font-size: 14px; border: 1px solid {C['border']}; padding: 10px; color: {C['text1']}; }}")
        qa_grid = QGridLayout()
        
        self.lbl_toplam_islem = QLabel("Toplam İşlem: --")
        self.lbl_win_rate_qa = QLabel("İsabet Oranı (Win Rate): --")
        self.lbl_profit_factor = QLabel("Kâr Faktörü: --")
        self.lbl_avg_win = QLabel("Ortalama Kâr: --")
        self.lbl_avg_loss = QLabel("Ortalama Zarar: --")
        self.lbl_beklenen_deger = QLabel("Beklenen Değer: --")
        
        self.lbl_mc_drawdown = QLabel("MC %99 Max DD: --")
        self.lbl_mc_iflas = QLabel("İflas Riski (%50 DD): --")
        self.lbl_mc_median = QLabel("MC Medyan P&L: --")

        font_qa = QFont(SANS, 11)
        for lbl in [self.lbl_toplam_islem, self.lbl_win_rate_qa, self.lbl_profit_factor, self.lbl_avg_win, self.lbl_avg_loss, self.lbl_beklenen_deger, self.lbl_mc_drawdown, self.lbl_mc_iflas, self.lbl_mc_median]:
            lbl.setFont(font_qa)
            lbl.setStyleSheet(f"color: {C['text1']};")

        qa_grid.addWidget(self.lbl_toplam_islem, 0, 0)
        qa_grid.addWidget(self.lbl_win_rate_qa, 0, 1)
        qa_grid.addWidget(self.lbl_profit_factor, 1, 0)
        qa_grid.addWidget(self.lbl_beklenen_deger, 1, 1)
        qa_grid.addWidget(self.lbl_avg_win, 2, 0)
        qa_grid.addWidget(self.lbl_avg_loss, 2, 1)
        qa_grid.addWidget(self.lbl_mc_drawdown, 3, 0)
        qa_grid.addWidget(self.lbl_mc_iflas, 3, 1)
        qa_grid.addWidget(self.lbl_mc_median, 4, 0)
        
        qa_grup.setLayout(qa_grid)
        lay.addWidget(qa_grup)

        lbl_gecmis = QLabel("📋 Kapanan İşlemler ve Giriş Nedenleri")
        lbl_gecmis.setFont(QFont(SANS, 12, QFont.Weight.Bold))
        lbl_gecmis.setStyleSheet(f"color: {C['text1']};")
        lay.addWidget(lbl_gecmis)

        self.gecmis_tablosu_qa = QTableWidget(0, 9)
        self.gecmis_tablosu_qa.setHorizontalHeaderLabels(["Sembol", "Yön", "Adet", "Giriş Fiyatı", "Çıkış Fiyatı", "P&L ($)", "P&L (%)", "Kapanış Sebebi", "Giriş Nedenleri"])
        
        header = self.gecmis_tablosu_qa.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        
        self.gecmis_tablosu_qa.setStyleSheet(f"""
            QTableWidget {{ background-color: {C['bg']}; color: {C['text1']}; border: 1px solid {C['border']}; gridline-color: {C['border']}; }}
            QHeaderView::section {{ background-color: {C['bg3']}; padding: 4px; border: 1px solid {C['border']}; font-weight: bold; color: {C['text3']}; }}
        """)
        lay.addWidget(self.gecmis_tablosu_qa)
        
        return w

    def _sistem_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(20)
        
        title = section_title("SİSTEM KONTROL PANELİ")
        title.setFont(QFont(SANS, 14, QFont.Weight.Bold))
        lay.addWidget(title)
        
        info = label("Bot motorlarını başlatın, eğitim durumunu kontrol edin ve çalışma modunu ayarlayın.", color=C["text2"])
        lay.addWidget(info)
        
        lay.addSpacing(20)
        
        btn_area = QWidget()
        btn_area.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(btn_area)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(15)

        def ctrl_btn(text, bg_hex, border_hex, color_hex, komut, run_cb=None):
            b = QPushButton(text)
            b.setFont(QFont(MONO, 12, QFont.Weight.Bold))
            b.setFixedHeight(45)
            
            style_template = f"""
                QPushButton {{
                    background: {bg_hex};
                    color: {color_hex};
                    border: 1px solid {border_hex};
                    border-radius: 8px;
                    padding: 0 15px;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {bg_hex.replace('12', '22').replace('0d', '1a').replace('2e', '40')};
                }}
                QPushButton:disabled {{
                    opacity: 0.4;
                }}
            """
            b.setStyleSheet(style_template)
            
            # Save original properties to restore later
            b.orj_text = text
            b.orj_style = style_template
            
            b.komut = komut
            b.run_cb = run_cb
            
            b.clicked.connect(lambda: self._toggle_motor(b))
            return b

        self.btn_canli  = ctrl_btn("▶  CANLI BAŞLAT",  "#0d2e25", "#1a5c45", C["accent"], [sys.executable, "-u", "proje2.py", "LIVE"], self._setup_canli)
        self.btn_sanal  = ctrl_btn("◆  SANAL TEST",    "#1c1040", "#3d2a7a", C["purple"], [sys.executable, "-u", "proje2.py", "SANAL_TEST"], self._setup_sanal)
        self.btn_zaman  = ctrl_btn("◷  ZAMAN MAKİNESİ","#2b1f0a", "#5c4010", C["amber"],  [sys.executable, "-u", "proje2.py", "ZAMAN_MAKINASI"], self._setup_zaman)
        self.btn_egitim = ctrl_btn("◎  BEYİN EĞİT",   "#0a1e30", "#1a4060", C["blue"],   [sys.executable, "-u", "egitim_dongusu.py"], None)

        self.sistem_butonlari = [self.btn_canli, self.btn_sanal, self.btn_zaman, self.btn_egitim]
        self.aktif_motor_btn = None
        
        for b in self.sistem_butonlari:
            bl.addWidget(b)

        lay.addWidget(btn_area)
        lay.addStretch()
        return w

    def _portfoy_card(self):
        c = card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        h = QFrame()
        h.setFixedHeight(42)
        h.setStyleSheet(f"background:transparent; border-bottom:1px solid {C['border']}; border-radius:0px;")
        hl = QHBoxLayout(h)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(section_title("AKTİF PORTFÖY"))
        hl.addStretch()
        self.badge_portfoy = QLabel("0 POZİSYON")
        self.badge_portfoy.setFont(QFont(MONO, 8, QFont.Weight.Bold))
        self.badge_portfoy.setStyleSheet(f"color:{C['accent']}; background:rgba(0,212,170,.1); border:1px solid rgba(0,212,170,.3); border-radius:8px; padding:2px 8px;")
        hl.addWidget(self.badge_portfoy)
        lay.addWidget(h)

        self.tablo_portfoy = build_table(["Sembol", "Yön", "Adet", "Giriş", "Stop", "AI Skor"])
        lay.addWidget(self.tablo_portfoy)
        return c

    def _on_combo_sembol_degisti(self, sembol_text):
        if not sembol_text:
            self.mum_grafik_widget.getPlotItem().clear()
            return
            
        sembol = sembol_text.split(" ")[0]
        
        # Portfoy dosyasından giriş ve stop seviyelerini oku
        giris = 0.0
        stop = 0.0
        yon = "LONG"
        try:
            p_dosya = CANLI_PORTFOY if self.gosterim_modu == "CANLI" else SANAL_PORTFOY
            if os.path.exists(p_dosya):
                with open(p_dosya, 'r', encoding='utf-8') as f:
                    portfoy = json.load(f)
                    if sembol in portfoy:
                        giris = portfoy[sembol].get("Fiyat", 0.0)
                        stop = portfoy[sembol].get("Stop", 0.0)
                        yon = portfoy[sembol].get("Yon", "LONG")
        except:
            pass
            
        self.mum_grafik_ciz(sembol, yon, giris, stop)

    def mum_grafik_ciz(self, sembol, yon, giris, stop):
        self.mum_grafik_widget.getPlotItem().clear()
        
        # yfinance'dan veri çekelim
        try:
            ticker = yf.Ticker(sembol)
            df = ticker.history(period="1mo")
            if df.empty:
                return
            
            # DataFrame'i list of tuples formatına dönüştür: (t, open, close, min, max)
            data = []
            for i, (idx, row) in enumerate(df.iterrows()):
                data.append((i, row['Open'], row['Close'], row['Low'], row['High']))
                
            # Mum grafiğini çiz
            item = CandlestickItem(data)
            self.mum_grafik_widget.addItem(item)
            
            # X ekseninde tarihleri göstermek için
            ticks = [(i, idx.strftime("%d %b")) for i, idx in enumerate(df.index) if i % 5 == 0]
            self.mum_grafik_widget.getPlotItem().getAxis('bottom').setTicks([ticks])
            
            # Yatay çizgileri çiz
            if giris > 0:
                self.mum_grafik_widget.addLine(y=giris, pen=pg.mkPen(C["blue"], width=2, style=Qt.PenStyle.DashLine))
            if stop > 0:
                self.mum_grafik_widget.addLine(y=stop, pen=pg.mkPen(C["red"], width=2, style=Qt.PenStyle.DotLine))
                
            # Take-Profit (Hedef) çizgisini hesapla: (Giriş ile Stop arasındaki mesafenin 1.5 katı)
            if giris > 0 and stop > 0:
                mesafe = abs(giris - stop)
                hedef = giris + (mesafe * 1.5) if yon == "LONG" else giris - (mesafe * 1.5)
                self.mum_grafik_widget.addLine(y=hedef, pen=pg.mkPen(C["accent"], width=2, style=Qt.PenStyle.DotLine))

        except Exception as e:
            print(f"Grafik çizim hatası: {e}")

    def _gecmis_card(self):
        c = card()
        lay = QVBoxLayout(c)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        h = QFrame()
        h.setFixedHeight(42)
        h.setStyleSheet(f"background:transparent; border-bottom:1px solid {C['border']}; border-radius:0px;")
        hl = QHBoxLayout(h)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(section_title("SON İŞLEMLER"))
        hl.addStretch()
        lay.addWidget(h)

        self.tablo_gecmis = build_table(["Zaman", "Sembol", "Yön", "Adet", "Giriş F.", "Çıkış F.", "P&L ($)", "Sebep"])
        lay.addWidget(self.tablo_gecmis)
        return c

    # ── SAĞ PANEL ─────────────────────────────────────────────
    def _right_panel(self):
        rp = QFrame()
        rp.setFixedWidth(272)
        rp.setStyleSheet(f"""
            QFrame {{
                background: {C['bg2']};
                border-left: 1px solid {C['border']};
                border-radius: 0px;
            }}
        """)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ background: {C['bg2']}; width: 4px; }}
            QScrollBar::handle:vertical {{ background: {C['border2']}; border-radius: 2px; }}
        """)

        content = QWidget()
        content.setStyleSheet(f"background: {C['bg2']};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        def rp_section(title):
            sec = QFrame()
            sec.setStyleSheet(f"QFrame {{ background: transparent; border-bottom: 1px solid {C['border']}; border-radius: 0px; }}")
            sl = QVBoxLayout(sec)
            sl.setContentsMargins(14, 12, 14, 12)
            sl.setSpacing(8)
            t = section_title(title)
            sl.addWidget(t)
            return sec, sl

        # Makro Göstergeler
        sec, sl = rp_section("MAKRO GÖSTERGELER")
        for attr, isim, renk, pct in [
            ("lbl_vix", "VIX (Korku)", C["amber"], 36),
            ("lbl_dxy", "DXY (Dolar)", C["blue"],  62),
        ]:
            row = QHBoxLayout()
            row.setSpacing(0)
            name_lbl = label(isim, color=C["text2"], size=10, mono=True)
            val_lbl  = label("--",  color=renk,       size=11, bold=True, mono=True)
            setattr(self, attr, val_lbl)
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(val_lbl)
            sl.addLayout(row)
            bar_bg = QFrame()
            bar_bg.setFixedHeight(3)
            bar_bg.setStyleSheet(f"background:{C['bg4']}; border-radius:1px; border:none;")
            bar_fill = QFrame(bar_bg)
            bar_fill.setFixedHeight(3)
            bar_fill.setStyleSheet(f"background:{renk}; border-radius:1px; border:none;")
            if attr == "lbl_vix":
                self._vix_bar = bar_fill
            else:
                self._dxy_bar = bar_fill
            bar_fill.setFixedWidth(int(bar_bg.sizeHint().width() * pct / 100))
            sl.addWidget(bar_bg)
        cl.addWidget(sec)

        # Sektör Para Akışı
        sec2, sl2 = rp_section("SEKTÖR PARA AKIŞI (5G %)")
        self.sektor_karti = []
        sektor_verileri = [
            ("Teknoloji", 2.14), ("Finans",   0.83),
            ("Sağlık",   -0.41), ("Enerji",   1.27),
            ("Sanayi",   -0.18), ("Defansif", 0.55),
            ("Hammadde",  0.12), ("Emtia",    0.87),
        ]
        grid = QGridLayout()
        grid.setSpacing(5)
        for i, (isim, pct) in enumerate(sektor_verileri):
            k = SektorKart(isim, pct)
            k.setFixedHeight(52)
            grid.addWidget(k, i // 2, i % 2)
            self.sektor_karti.append(k)
        sl2.addLayout(grid)
        cl.addWidget(sec2)

        # Beyin Eğitim Durumu
        sec3, sl3 = rp_section("UZMAN BEYİN DURUMU")
        beyin_verileri = [
            ("Teknoloji", 100, 100), ("Finans",    100, 100),
            ("Sağlık",     67, 100), ("Enerji",     34, 100),
            ("Sanayi",    100, 100), ("Defansif",  100, 100),
            ("Emtia",     100, 100),
        ]
        self.beyin_satirlari = []
        for isim, ep, top in beyin_verileri:
            row = BeynilerlemeRow(isim, ep, top)
            sl3.addWidget(row)
            self.beyin_satirlari.append((isim, row))
        cl.addWidget(sec3)

        # Log Terminali
        sec4, sl4 = rp_section("SİSTEM LOGLARI")
        self.log_terminali = LogTerminal()
        self.log_terminali.setMinimumHeight(200)
        self.log_terminali.log("Sistem başlatıldı — emirleriniz bekleniyor.", "sys")
        sl4.addWidget(self.log_terminali)
        cl.addWidget(sec4)
        cl.addStretch()

        scroll.setWidget(content)
        rp_outer = QFrame()
        rp_outer.setFixedWidth(272)
        rp_outer.setStyleSheet(f"background:{C['bg2']}; border-left:1px solid {C['border']}; border-radius:0px;")
        ol = QVBoxLayout(rp_outer)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(scroll)
        return rp_outer

    # ─────────────────────────────────────────────────────────
    # MOTOR KONTROLÜ VE TOGGLE YAPISI
    # ─────────────────────────────────────────────────────────
    def _toggle_motor(self, btn):
        if self.aktif_motor is not None:
            # Motor zaten çalışıyor. Tıklanan buton aktif buton ise durdur.
            if self.aktif_motor_btn == btn:
                self.motoru_durdur()
            else:
                self.log_terminali.log("Zaten çalışan bir motor var — önce onu durdurun.", "warn")
        else:
            # Motor çalışmıyor, başlat.
            if btn.run_cb:
                btn.run_cb()
            self.motoru_baslat(btn.komut, btn)

    def _setup_canli(self):
        self.gosterim_modu = "CANLI"
        self.lbl_status.setText("CANLI MOD")

    def _setup_sanal(self):
        self.gosterim_modu = "SANAL"
        self.lbl_status.setText("SANAL MOD")

    def _setup_zaman(self):
        self.gosterim_modu = "SANAL"
        self.lbl_status.setText("SANAL MOD (ZAMAN MAKİNESİ)")
        for dosya, icerik in [
            (SANAL_HESAP,   {"Baslangic_Bakiyesi": 10000.0, "Guncel_Bakiye": 10000.0, "Toplam_Kar_Zarar": 0.0}),
            (SANAL_PORTFOY, {}),
        ]:
            with open(dosya, 'w', encoding='utf-8') as f:
                json.dump(icerik, f)
        if os.path.exists(SANAL_GECMIS):
            os.remove(SANAL_GECMIS)

    def motoru_baslat(self, komut, btn):
        self.log_terminali.log(f"🚀 Başlatılıyor: {' '.join(komut)}", "info")
        
        # Diğer butonları devre dışı bırak
        for b in self.sistem_butonlari:
            if b != btn:
                b.setEnabled(False)
                
        # Aktif butonu DURDUR stiline çevir
        btn.setText("■  DURDUR")
        btn.setStyleSheet(f"""
            QPushButton {{
                background: #2e0d15;
                color: {C['red']};
                border: 1px solid #5c1a27;
                border-radius: 8px;
                padding: 0 15px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: #40101d;
            }}
        """)
        self.aktif_motor_btn = btn

        self.aktif_motor = BotMotoru(komut)
        self.aktif_motor.log_sinyali.connect(self._motor_log)
        self.aktif_motor.bitis_sinyali.connect(self.motor_bitti)
        self.aktif_motor.start()

    def _motor_log(self, mesaj):
        self.log_terminali.log(mesaj)

    def motor_bitti(self):
        self.log_terminali.log("Motor durdu / tamamlandı.", "sys")
        
        if self.aktif_motor_btn:
            # Butonu eski haline getir
            self.aktif_motor_btn.setText(self.aktif_motor_btn.orj_text)
            self.aktif_motor_btn.setStyleSheet(self.aktif_motor_btn.orj_style)
            
        for b in self.sistem_butonlari:
            b.setEnabled(True)
            
        self.aktif_motor_btn = None
        self.aktif_motor = None

    def motoru_durdur(self):
        if self.aktif_motor:
            self.log_terminali.log("Motor zorla durduruluyor...", "warn")
            self.aktif_motor.durdur()

    # ─────────────────────────────────────────────────────────
    # VERİ GÜNCELLEME (her 1 sn)
    # ─────────────────────────────────────────────────────────
    def verileri_guncelle(self):
        if self.gosterim_modu == "CANLI":
            c_pnl = self._ortam_guncelle(
                CANLI_HESAP, CANLI_PORTFOY, CANLI_GECMIS,
                self.tablo_portfoy, self.tablo_gecmis,
                prefix="canli", is_primary=True
            )
            s_pnl = self._ortam_guncelle(
                SANAL_HESAP, SANAL_PORTFOY, SANAL_GECMIS,
                None, None,
                prefix="sanal", is_primary=False
            )
        else:
            c_pnl = self._ortam_guncelle(
                CANLI_HESAP, CANLI_PORTFOY, CANLI_GECMIS,
                None, None,
                prefix="canli", is_primary=False
            )
            s_pnl = self._ortam_guncelle(
                SANAL_HESAP, SANAL_PORTFOY, SANAL_GECMIS,
                self.tablo_portfoy, self.tablo_gecmis,
                prefix="sanal", is_primary=True
            )

        # Grafik güncelle
        if self.aktif_motor is not None:
            c = c_pnl if c_pnl is not None else (self.y_canli_pnl[-1] if self.y_canli_pnl else 0.0)
            s = s_pnl if s_pnl is not None else (self.y_sanal_pnl[-1] if self.y_sanal_pnl else 0.0)
            self.pnl_grafik.guncelle(c, s)
            self.y_canli_pnl.append(c)
            self.y_sanal_pnl.append(s)

        self._makro_sayac += 1
        # Her 30 saniyede bir (ilk saniyede dahil) makro veritabanından çek
        if self._makro_sayac % 30 == 1:
            self._makro_guncelle()

    def _makro_guncelle(self):
        import sqlite3
        db_yolu = "yapay_zeka_veritabani.sqlite"
        if not os.path.exists(db_yolu): return
        
        sektor_etfleri = {
            'Teknoloji': 'XLK', 'Finans': 'XLF', 'Sağlık': 'XLV',
            'Enerji': 'XLE', 'Sanayi': 'XLI', 'Defansif': 'XLP',
            'Hammadde': 'XLB'
        }
        
        try:
            with sqlite3.connect(db_yolu) as conn:
                cur = conn.cursor()
                
                # Sektör verileri (Son 6 günlük 1d kapanışlarına göre 5-günlük getiri)
                for i, (isim, k_widget) in enumerate(zip(["Teknoloji", "Finans", "Sağlık", "Enerji", "Sanayi", "Defansif", "Hammadde", "Emtia"], self.sektor_karti)):
                    if isim == "Emtia": continue # Emtia ETF'si veritabanında cache'de yoksa atla
                    etf = sektor_etfleri.get(isim)
                    if not etf: continue
                    
                    try:
                        # SQLite cache tablosundan en güncel 6 günü çek
                        cur.execute(f'SELECT Close FROM "cache_{etf}_1d" ORDER BY Datetime DESC LIMIT 6')
                        satirlar = cur.fetchall()
                        if len(satirlar) >= 6:
                            son = satirlar[0][0]
                            eski = satirlar[5][0]
                            pct = ((son - eski) / eski) * 100.0
                            k_widget.guncelle(pct)
                    except Exception:
                        pass
                
                # VIX (Korku Endeksi)
                try:
                    cur.execute('SELECT Close FROM "cache__VIX_1d" ORDER BY Datetime DESC LIMIT 1')
                    vix = cur.fetchone()
                    if vix:
                        vix_val = vix[0]
                        self.lbl_vix.setText(f"{vix_val:.2f}")
                        # 0-50 arasını %0-100 barına çevir
                        oran = min(vix_val, 50.0) / 50.0
                        self._vix_bar.setFixedWidth(max(0, int(self._vix_bar.parent().width() * oran)))
                except Exception: pass
                
                # DXY (Dolar Endeksi)
                try:
                    cur.execute('SELECT Close FROM "cache_DX_Y_NYB_1d" ORDER BY Datetime DESC LIMIT 1')
                    dxy = cur.fetchone()
                    if dxy:
                        dxy_val = dxy[0]
                        self.lbl_dxy.setText(f"{dxy_val:.2f}")
                        # 90-110 arasını bar için ölçekle (veya 0-120)
                        oran = min(dxy_val, 120.0) / 120.0
                        self._dxy_bar.setFixedWidth(max(0, int(self._dxy_bar.parent().width() * oran)))
                except Exception: pass
        except Exception:
            pass

        # Beyin durumlarını (PTH dosyaları) kontrol et
        try:
            cwd = os.path.dirname(os.path.abspath(__file__))
            for isim, row_widget in getattr(self, "beyin_satirlari", []):
                dosya_isim = isim.replace(" ", "_").replace("ğ", "g").replace("ı", "i").upper()
                if isim == "Emtia": dosya_isim = "EMTIA_VE_MADENCILIK"
                elif isim == "Defansif": dosya_isim = "TUKETICI_TEMEL_IHTIYAC"
                
                long_pth = os.path.join(cwd, f"{dosya_isim}_long_hiyerarsik_beyin.pth")
                short_pth = os.path.join(cwd, f"{dosya_isim}_short_hiyerarsik_beyin.pth")
                
                is_ready = os.path.exists(long_pth) and os.path.exists(short_pth)
                row_widget.guncelle_durum(is_ready)
        except Exception:
            pass

    def _ortam_guncelle(self, h_dosya, p_dosya, g_dosya, tablo_portfoy, tablo_gecmis, prefix="canli", is_primary=False):
        pnl = None

        def dosya_degisti_mi(dosya_yolu):
            if not os.path.exists(dosya_yolu):
                return False
            guncel_mtime = os.path.getmtime(dosya_yolu)
            eski_mtime = getattr(self, '_mtimes', {}).get(dosya_yolu, 0)
            if guncel_mtime > eski_mtime:
                if not hasattr(self, '_mtimes'):
                    self._mtimes = {}
                self._mtimes[dosya_yolu] = guncel_mtime
                return True
            return False

        # Hesap
        try:
            if os.path.exists(h_dosya):
                with open(h_dosya, 'r', encoding='utf-8') as f:
                    hesap = json.load(f)
                bakiye = hesap.get("Guncel_Bakiye", 0.0)
                toplam_pnl = hesap.get("Toplam_Kar_Zarar", 0.0)
                pnl = toplam_pnl
                setattr(self, f'_{prefix}_son_pnl', pnl)

                if is_primary:
                    sign  = "+" if toplam_pnl >= 0 else "-" if toplam_pnl < 0 else ""
                    renk  = C["accent"] if toplam_pnl >= 0 else C["red"]
                    self.lbl_top_bakiye.setText(f"${bakiye:,.2f}")
                    self.lbl_top_pnl.setText(f"{sign}${abs(toplam_pnl):,.2f}")
                    self.lbl_top_pnl.setStyleSheet(f"color:{renk}; background:transparent; border:none;")
                    self.kpi_pnl.set_value(f"{sign}${abs(toplam_pnl):,.2f}", color=renk)
                    
                    if hasattr(self, 'lbl_top_bugun'):
                        gunluk_baslangic = hesap.get("Gun_Baslangic_Bakiyesi", bakiye)
                        gunluk_pnl = bakiye - gunluk_baslangic
                        sign_gun = "+" if gunluk_pnl >= 0 else "-" if gunluk_pnl < 0 else ""
                        renk_gun = C["accent"] if gunluk_pnl >= 0 else C["red"]
                        gunluk_pnl_yuzde = (abs(gunluk_pnl) / gunluk_baslangic * 100) if gunluk_baslangic > 0 else 0
                        self.lbl_top_bugun.setText(f"{sign_gun}${abs(gunluk_pnl):,.2f} ({sign_gun}%{gunluk_pnl_yuzde:.2f})")
                        self.lbl_top_bugun.setStyleSheet(f"color:{renk_gun}; background:transparent; border:none;")
        except Exception as e:
            print(f"[ARAYÜZ HATA] Hesap güncellenemedi ({h_dosya}): {e}")

        # Portföy
        if tablo_portfoy:
            try:
                if os.path.exists(p_dosya):
                    with open(p_dosya, 'r', encoding='utf-8') as f:
                        portfoy = json.load(f)
                    tablo_portfoy.setRowCount(len(portfoy))
                    self.badge_portfoy.setText(f"{len(portfoy)} POZİSYON")
                    long_c = short_c = 0
                    
                    if hasattr(self, 'combo_sembol'):
                        mevcut_semboller = [self.combo_sembol.itemText(j).split(" ")[0] for j in range(self.combo_sembol.count())]
                        yeni_semboller = list(portfoy.keys())
                        for s in yeni_semboller:
                            if s not in mevcut_semboller:
                                y = portfoy[s].get("Yon", "")
                                self.combo_sembol.addItem(f"{s} ({y})")
                        for j in range(self.combo_sembol.count() - 1, -1, -1):
                            if self.combo_sembol.itemText(j).split(" ")[0] not in yeni_semboller:
                                self.combo_sembol.removeItem(j)
                                
                    for i, (sembol, veri) in enumerate(portfoy.items()):
                        yon   = veri.get("Yon", "LONG")
                        fiyat = veri.get("Fiyat", 0)
                        stop  = veri.get("Stop", 0)
                        skor  = veri.get("Skor", 0)
                        adet  = veri.get("Adet", 0)
                        
                        if yon == "LONG": long_c += 1
                        else: short_c += 1

                        def item(txt, color=None, align=Qt.AlignmentFlag.AlignVCenter):
                            it = QTableWidgetItem(txt)
                            if color:
                                it.setForeground(QColor(color))
                            it.setTextAlignment(align | Qt.AlignmentFlag.AlignLeft)
                            return it

                        tablo_portfoy.setItem(i, 0, item(sembol))
                        yon_item = QTableWidgetItem(yon)
                        yon_item.setForeground(QColor(C["accent"] if yon == "LONG" else C["red"]))
                        tablo_portfoy.setItem(i, 1, yon_item)
                        tablo_portfoy.setItem(i, 2, item(f"{adet}"))
                        tablo_portfoy.setItem(i, 3, item(f"${fiyat:.2f}"))
                        tablo_portfoy.setItem(i, 4, item(f"${stop:.2f}", color=C["red"]))
                        tablo_portfoy.setItem(i, 5, item(f"%{skor:.0f}", color=C["blue"]))

                    self.kpi_pos.set_value(str(len(portfoy)),
                        sub=f"{long_c} LONG · {short_c} SHORT")
                else:
                    tablo_portfoy.setRowCount(0)
            except Exception:
                pass

        # Geçmiş
        if tablo_gecmis:
            try:
                if os.path.exists(g_dosya):
                    with open(g_dosya, 'r', encoding='utf-8') as f:
                        satirlar = list(csv.reader(f))
                    if len(satirlar) > 1:
                        gecmis = list(reversed(satirlar[1:]))[:15]
                        tablo_gecmis.setRowCount(len(gecmis))
                        kazanma = sum(1 for r in satirlar[1:] if len(r) > 6 and self._safe_float(r[6]) > 0)
                        toplam  = len(satirlar) - 1
                        if toplam > 0:
                            wr = kazanma / toplam * 100
                            self.kpi_wr.set_value(f"{wr:.1f}%",
                                sub=f"{toplam} işlemden {kazanma} kâr",
                                color=C["accent"] if wr >= 50 else C["red"])
                                
                            # --- QA İstatistikleri Güncelleme ---
                            toplam_kar = sum(self._safe_float(r[6]) for r in satirlar[1:] if len(r) > 6 and self._safe_float(r[6]) > 0)
                            toplam_zarar = sum(abs(self._safe_float(r[6])) for r in satirlar[1:] if len(r) > 6 and self._safe_float(r[6]) < 0)
                            profit_factor = (toplam_kar / toplam_zarar) if toplam_zarar > 0 else (toplam_kar if toplam_kar > 0 else 0)
                            avg_win = (toplam_kar / kazanma) if kazanma > 0 else 0
                            kaybetme = toplam - kazanma
                            avg_loss = (toplam_zarar / kaybetme) if kaybetme > 0 else 0
                            beklenen_deger = (avg_win * (kazanma/toplam)) - (avg_loss * (kaybetme/toplam))
                            
                            if hasattr(self, 'lbl_toplam_islem'):
                                self.lbl_toplam_islem.setText(f"Toplam İşlem: {toplam}")
                                self.lbl_win_rate_qa.setText(f"İsabet Oranı (Win Rate): %{wr:.1f}")
                                self.lbl_profit_factor.setText(f"Kâr Faktörü: {profit_factor:.2f}")
                                self.lbl_avg_win.setText(f"Ortalama Kâr: ${avg_win:.2f}")
                                self.lbl_avg_loss.setText(f"Ortalama Zarar: ${avg_loss:.2f}")
                                self.lbl_beklenen_deger.setText(f"Beklenen Değer: ${beklenen_deger:.2f}")

                                # --- MONTE CARLO SIMULATION ---
                                islem_sonuclari = [self._safe_float(r[7]) for r in satirlar[1:] if len(r) > 7]
                                if len(islem_sonuclari) >= 10:
                                    if not hasattr(self, '_mc_cache'): self._mc_cache = {}
                                    mc_key = f"{g_dosya}_{len(islem_sonuclari)}"
                                    
                                    if mc_key not in self._mc_cache:
                                        import numpy as np
                                        pnl_array = np.array(islem_sonuclari)
                                        simulasyon_sayisi = 10000
                                        islem_sayisi = 500
                                        
                                        rastgele_senaryolar = np.random.choice(pnl_array, size=(simulasyon_sayisi, islem_sayisi), replace=True)
                                        kasa_egrileri = np.cumsum(rastgele_senaryolar, axis=1)
                                        tepeler = np.maximum.accumulate(kasa_egrileri, axis=1)
                                        dususler = tepeler - kasa_egrileri
                                        max_drawdowns = np.max(dususler, axis=1)
                                        
                                        guvenli_drawdown = np.percentile(max_drawdowns, 99)
                                        median_pnl = np.median(kasa_egrileri[:, -1])
                                        iflas_sayisi = np.sum(max_drawdowns >= 50.0)
                                        iflas_riski = (iflas_sayisi / simulasyon_sayisi) * 100
                                        
                                        self._mc_cache[mc_key] = (guvenli_drawdown, median_pnl, iflas_riski)
                                    else:
                                        guvenli_drawdown, median_pnl, iflas_riski = self._mc_cache[mc_key]
                                        
                                    if hasattr(self, 'lbl_mc_drawdown'):
                                        self.lbl_mc_drawdown.setText(f"MC %99 Max DD: %{guvenli_drawdown:.2f}")
                                        self.lbl_mc_iflas.setText(f"İflas Riski (%50 DD): %{iflas_riski:.2f}")
                                        self.lbl_mc_median.setText(f"MC Medyan P&L: %{median_pnl:.2f}")
                                # ------------------------------

                                qa_gecmis = list(reversed(satirlar[1:]))[:50]
                                self.gecmis_tablosu_qa.setRowCount(len(qa_gecmis))
                                for j, r in enumerate(qa_gecmis):
                                    if len(r) < 7: continue
                                    sembol = r[1]
                                    yon = r[2] if len(r) > 2 else ""
                                    giris_f = self._safe_float(r[3]) if len(r) > 3 else 0.0
                                    cikis_f = self._safe_float(r[4]) if len(r) > 4 else 0.0
                                    adet = r[5] if len(r) > 5 else "0"
                                    pnl_usd = self._safe_float(r[6])
                                    pnl_yuzde = self._safe_float(r[7]) if len(r) > 7 else 0.0
                                    kapanis_sebebi = r[8] if len(r) > 8 else ""
                                    giris_nedenleri = r[9] if len(r) > 9 else "Bilinmiyor (Eski Kayıt)"
                                    
                                    it_sembol = QTableWidgetItem(sembol)
                                    it_sembol.setFont(QFont(MONO, 10, QFont.Weight.Bold))
                                    
                                    it_yon = QTableWidgetItem(yon)
                                    it_yon.setForeground(QColor(C["accent"] if yon == "LONG" else C["red"]))
                                    
                                    it_pnl_usd = QTableWidgetItem(f"${pnl_usd:.2f}")
                                    it_pnl_yuzde = QTableWidgetItem(f"%{pnl_yuzde:.2f}")
                                    
                                    c_renk = C["accent"] if pnl_usd > 0 else C["red"]
                                    it_pnl_usd.setForeground(QColor(c_renk))
                                    it_pnl_yuzde.setForeground(QColor(c_renk))
                                    
                                    self.gecmis_tablosu_qa.setItem(j, 0, it_sembol)
                                    self.gecmis_tablosu_qa.setItem(j, 1, it_yon)
                                    self.gecmis_tablosu_qa.setItem(j, 2, QTableWidgetItem(adet))
                                    self.gecmis_tablosu_qa.setItem(j, 3, QTableWidgetItem(f"${giris_f:.2f}"))
                                    self.gecmis_tablosu_qa.setItem(j, 4, QTableWidgetItem(f"${cikis_f:.2f}"))
                                    self.gecmis_tablosu_qa.setItem(j, 5, it_pnl_usd)
                                    self.gecmis_tablosu_qa.setItem(j, 6, it_pnl_yuzde)
                                    self.gecmis_tablosu_qa.setItem(j, 7, QTableWidgetItem(kapanis_sebebi))
                                    self.gecmis_tablosu_qa.setItem(j, 8, QTableWidgetItem(giris_nedenleri))
                            # ------------------------------------

                        for i, row in enumerate(gecmis):
                            if len(row) < 7:
                                continue
                            saat = row[0].split()[1] if ' ' in row[0] else row[0]
                            giris_f = self._safe_float(row[3]) if len(row) > 3 else 0.0
                            cikis_f = self._safe_float(row[4]) if len(row) > 4 else 0.0
                            adet = row[5] if len(row) > 5 else "0"
                            pnl_val = self._safe_float(row[6])
                            sign  = "+" if pnl_val >= 0 else ""
                            renk  = C["accent"] if pnl_val >= 0 else C["red"]
                            sebep = row[8] if len(row) > 8 else ""

                            def it(t, c=None):
                                x = QTableWidgetItem(t)
                                if c: x.setForeground(QColor(c))
                                return x

                            tablo_gecmis.setItem(i, 0, it(saat,  C["text3"]))
                            tablo_gecmis.setItem(i, 1, it(row[1]))
                            yon_it = QTableWidgetItem(row[2])
                            yon_it.setForeground(QColor(C["accent"] if row[2]=="LONG" else C["red"]))
                            tablo_gecmis.setItem(i, 2, yon_it)
                            tablo_gecmis.setItem(i, 3, it(adet))
                            tablo_gecmis.setItem(i, 4, it(f"${giris_f:.2f}"))
                            tablo_gecmis.setItem(i, 5, it(f"${cikis_f:.2f}"))
                            tablo_gecmis.setItem(i, 6, it(f"{sign}${abs(pnl_val):.2f}", renk))
                            tablo_gecmis.setItem(i, 7, it(sebep, C["text3"]))
                else:
                    tablo_gecmis.setRowCount(0)
            except Exception:
                pass

        return pnl

    @staticmethod
    def _safe_float(s):
        try: return float(s)
        except: return 0.0

    def closeEvent(self, event):
        if self.aktif_motor:
            self.aktif_motor.durdur()
            self.aktif_motor.wait(2000)
        event.accept()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(C["bg"]))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(C["text1"]))
    pal.setColor(QPalette.ColorRole.Base,            QColor(C["bg2"]))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(C["bg3"]))
    pal.setColor(QPalette.ColorRole.Text,            QColor(C["text1"]))
    pal.setColor(QPalette.ColorRole.Button,          QColor(C["bg3"]))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(C["text1"]))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(C["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    app.setPalette(pal)

    pencere = CommandCenter()
    pencere.show()
    sys.exit(app.exec())
