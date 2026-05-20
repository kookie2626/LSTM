import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# load_meter from train_all_meters
sys.path.append('/home/keun/workspace')
from project.ML.scripts.train.train_all_meters import load_meter

print("Fetching data for H1.ZE20...")
series = load_meter('H1.ZE20')
df = series.to_frame(name='P')

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
