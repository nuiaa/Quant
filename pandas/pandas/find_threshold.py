with open('proje2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
keywords = ['0.5', '0.6', 'esik', 'esigi', 'ESIK', 'threshold', 'AI_Puan', 'model(', 'olasilik', 'DinamikHiyerarsik', 'scaler_makro', 'scaler_teknik']
print("=== THRESHOLD / MODEL USAGE LINES ===")
for i, line in enumerate(lines, 1):
    if any(k in line for k in keywords):
        print(f'{i}: {line.rstrip()}')
