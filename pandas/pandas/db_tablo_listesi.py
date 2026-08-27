import sqlite3
conn = sqlite3.connect('yapay_zeka_veritabani.sqlite')
sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
tables = [r[0] for r in conn.execute(sql).fetchall()]
conn.close()
print(f'Toplam tablo: {len(tables)}')
for t in tables:
    print(t)
