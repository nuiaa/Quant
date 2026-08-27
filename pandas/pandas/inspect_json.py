import json
d = json.load(open('piyasa_haritasi.json','r',encoding='utf-8'))
sek = d['SEKTORLER']
print('Sektor sayisi:', len(sek))
for k, v in list(sek.items())[:3]:
    print(f'  Sektor: {k}')
    print(f'  Anahtarlar: {list(v.keys())}')
    if 'Endustriler' in v:
        for ek, ev in list(v['Endustriler'].items())[:1]:
            hisseler = ev.get('Hisseler', [])
            print(f'    Endustri: {ek}, Hisseler_ilk_5: {hisseler[:5]}')
    print()
