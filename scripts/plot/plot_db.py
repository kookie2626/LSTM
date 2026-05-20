import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import psycopg
from dotenv import load_dotenv

load_dotenv('/home/keun/workspace/project/ML/.env')

CONNECT_KWARGS = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ["DB_NAME"],
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

meter_urn = 'H1.ZE20'
sql = f"""
    SELECT ts, value
    FROM ems.cr_measurement_1h
    WHERE meter_urn = '{meter_urn}' AND measurement = 'P'
    ORDER BY ts
"""
with psycopg.connect(**CONNECT_KWARGS) as conn:
    df = pd.read_sql(sql, conn)

df.index = pd.to_datetime(df["ts"], utc=True)
df = df.rename(columns={'value': 'P'})

df_2022 = df.loc['2022-01-01':'2022-12-31']
df_2023 = df.loc['2023-01-01':'2023-12-31']

plt.figure(figsize=(15, 6))
plt.plot(df_2022.index, df_2022['P'], label='2022 (Validation)', color='#2c7bb6', alpha=0.8)
plt.plot(df_2023.index, df_2023['P'], label='2023 (Test)', color='#d7191c', alpha=0.8)
plt.title('Meter H1.ZE20 Power Consumption (2022 vs 2023)', fontsize=15, fontweight='bold')
plt.ylabel('Power (W)')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

art_dir = '/home/keun/.gemini/antigravity/brain/f323c41e-9aee-4de9-b081-8592b3b088cd/artifacts'
os.makedirs(art_dir, exist_ok=True)
plt.savefig(f'{art_dir}/H1_ZE20_usage.png', dpi=150)
print('Plot generated successfully')
