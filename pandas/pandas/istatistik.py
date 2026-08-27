import pandas as pd

def analiz_et():
    try:
        df = pd.read_csv('sanal_islem_gecmisi.csv')
    except Exception as e:
        print("CSV okunamadi:", e)
        return
        
    df['Giris_Tarihi'] = pd.to_datetime(df['Giris_Tarihi'])
    df['Cikis_Tarihi'] = pd.to_datetime(df['Cikis_Tarihi'])
    
    if df.empty:
        print("İşlem yok.")
        return
        
    min_date = df['Giris_Tarihi'].min()
    max_date = df['Cikis_Tarihi'].max()
    total_days = (max_date - min_date).days
    
    # Calculate average hold time
    avg_hold = df['Elde_Tutma_Gun'].mean()
    
    # Calculate days between trades
    df_sorted = df.sort_values('Giris_Tarihi')
    diffs = df_sorted['Giris_Tarihi'].diff().dt.days.dropna()
    avg_diff = diffs.mean() if not diffs.empty else 0
    max_diff = diffs.max() if not diffs.empty else 0
    
    trades_per_month = len(df) / (total_days / 30.44) if total_days > 0 else 0
    
    # Calculate active days (where at least one position was held)
    active_days = set()
    for _, row in df.iterrows():
        # Add all days between Giris and Cikis
        date_range = pd.date_range(start=row['Giris_Tarihi'], end=row['Cikis_Tarihi'])
        for d in date_range:
            active_days.add(d)
            
    active_days_count = len(active_days)
    activity_ratio = (active_days_count / total_days * 100) if total_days > 0 else 0
    
    print(f"Toplam Test Süresi: {total_days} gün")
    print(f"Toplam İşlem Sayısı: {len(df)}")
    print(f"Ortalama İşlemde Kalma Süresi: {avg_hold:.1f} gün")
    print(f"İşlem Sıklığı (Ayda Ortalama): {trades_per_month:.2f} işlem")
    print(f"Ortalama İki İşlem Arası Bekleme: {avg_diff:.1f} gün")
    print(f"En Uzun İşlemsiz Dönem (Kuraklık): {max_diff:.0f} gün")
    print(f"Kasanın Pazarda Aktif Olduğu Gün Oranı: %{activity_ratio:.1f} ({active_days_count} gün)")
    
    # Win rate and ROI
    win_rate = (df['P&L_USD'] > 0).mean() * 100
    print(f"Kazanma Oranı (Win Rate): %{win_rate:.1f}")

if __name__ == '__main__':
    analiz_et()
