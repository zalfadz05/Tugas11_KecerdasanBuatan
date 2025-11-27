# ======================================================
# SEGMENTASI PELANGGAN MENGGUNAKAN KMEANS + TSNE
# Dataset: segmentasi.csv
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# ======================================================
# 1. LOAD DATASET
# ======================================================

df = pd.read_csv('segmentasi.csv')
print("=== 5 DATA TERATAS ===")
print(df.head())

print("\n=== INFO DATASET ===")
print(df.info())

# ======================================================
# 2. HANDLE MISSING VALUE
# ======================================================

print("\nJumlah missing value per kolom:")
print(df.isnull().sum())

# Drop jika masih wajar
df = df.dropna()
print("Total data setelah drop NA:", len(df))

# ======================================================
# 3. LABEL ENCODING (jika ada data kategorik)
# ======================================================

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

print("\n=== Dataset setelah Encoding ===")
print(df.head())

# ======================================================
# 4. NORMALISASI DATA
# ======================================================

scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# ======================================================
# 5. TENTUKAN JUMLAH CLUSTER MENGGUNAKAN METODE SIKU
# ======================================================

errors = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(scaled)
    errors.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), errors, marker='o')
plt.title("Metode Siku untuk Menentukan Cluster Optimal")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# ======================================================
# 6. KMEANS CLUSTERING
# ======================================================

# Misal hasil siku menunjukkan k=4 (bisa ubah sesuai grafik Anda)
k_optimal = 4
model = KMeans(n_clusters=k_optimal, random_state=22)
clusters = model.fit_predict(scaled)

df['Cluster'] = clusters

print("\n=== Contoh Data + Cluster ===")
print(df.head())

# ======================================================
# 7. VISUALISASI TSNE 2D
# ======================================================

tsne = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_result = tsne.fit_transform(scaled)

df_tsne = pd.DataFrame({
    'x': tsne_result[:, 0],
    'y': tsne_result[:, 1],
    'Cluster': clusters
})

plt.figure(figsize=(8, 6))
sb.scatterplot(data=df_tsne, x='x', y='y', hue='Cluster', palette='tab10', s=60)
plt.title("Visualisasi Clustering Menggunakan t-SNE")
plt.show()

# ======================================================
# 8. SIMPAN HASIL
# ======================================================

df.to_csv('hasil_segmentasi.csv', index=False)
print("\nFile hasil segmentasi tersimpan sebagai: hasil_segmentasi.csv")
