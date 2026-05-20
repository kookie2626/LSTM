import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('outputs/all_meters_results.csv')
df = df.dropna(subset=['test_mae'])
df = df[df['test_mae'] > 0]

art_dir = '/home/keun/.gemini/antigravity/brain/f323c41e-9aee-4de9-b081-8592b3b088cd/artifacts'
os.makedirs(art_dir, exist_ok=True)

# 1. Test MAE Distribution
plt.figure(figsize=(10, 5))
plt.hist(df['test_mae'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Test MAE Distribution across 80 Meters', fontsize=14, fontweight='bold')
plt.xlabel('Test MAE (W)', fontsize=12)
plt.ylabel('Number of Meters', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{art_dir}/mae_distribution.png', dpi=150)
plt.close()

# 2. Test WAPE Distribution (filtering out extremely large WAPEs for better visibility)
df_wape = df[df['test_wape'] < 500] # Cap for visualization if any absurd values
plt.figure(figsize=(10, 5))
plt.hist(df_wape['test_wape'], bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.title('Test WAPE Distribution across Meters (Capped at 500%)', fontsize=14, fontweight='bold')
plt.xlabel('Test WAPE (%)', fontsize=12)
plt.ylabel('Number of Meters', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{art_dir}/wape_distribution.png', dpi=150)
plt.close()

print('Distribution plots generated successfully.')
