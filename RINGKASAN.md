# Rangkuman Proyek — Pengelompokan Pelanggan (Pengayaan Presentasi)

## 1. Cerita satu paragraf
Data 37.500 pelanggan tanpa label ingin dibagi ke segmen bermakna. Karena
berbasis jarak, data di-standardize dulu, lalu diuji 3 representasi fitur
(scaled, PCA-8, embedding Autoencoder) dengan K-Means k=2 sebagai penentu
klaster, plus Hierarchical dan DBSCAN sebagai pembanding. Evaluasi di 12.500
data uji memakai Silhouette, Davies–Bouldin, dan Calinski–Harabasz. Menang:
S2 (K-Means + PCA-8). Hasilnya 2 segmen: Loyal High-value (~29%) vs
Promo-sensitive/low-engagement (~71%).

## 2. Data (hafal yang bold)
- **37.500 latih / 12.500 uji**, 13 kolom: 1 ID + **10 numerik** + 2 kategorikal.
- Numerik: umur, income, **tenure**, **monthly_spend**, **purchase_rate**,
  order_value, promo_usage_rate, return_rate, browse_minutes, support_contacts.
- Kategorikal: payment_channel (A/B/C), market_area (A/B/C).
- Missing 4–8% → imputasi **median** (numerik) / modus (kategorikal).
- Fakta kunci: **tenure bimodal** (50,9% = 1 bln vs 27,0% = 72 bln),
  **income skew** (median 12,7 jt vs max 495 jt), **~20% nilai NOL**,
  channel/area/umur **tidak membedakan** segmen.

## 3. Preprocessing (hafal urutan + alasan)
Imputasi → One-Hot → **StandardScaler** (fit hanya di train) → 16 fitur.
Alasan scaling: algoritma jarak; tanpa itu income (ratusan ribu) menenggelamkan
promo (0–1). Bukti: tabel before-after + mean≈0, std=1 + histogram income.

## 4. Metode (satu kalimat per metode)
- **K-Means (utama)**: partisi k centroid, minimizing WCSS; k dipilih via
  **Elbow + Silhouette** k=2..8 → **k=2** (silhouette 0,315; k=3 anjlok 0,13).
- **Agglomerative ward (pembanding)**: bottom-up merging, dendrogram subsample
  2.000 (full 37k O(n²) tidak feasible).
- **DBSCAN (pembanding)**: density-based, eps=1,2 dari k-distance graph,
  min_samples=10; menemukan noise (label −1) yang dikecualikan dari metrik.
- **PCA**: kompresi linear; PCA-8 untuk S2, PCA-2 untuk visualisasi.
- **Autoencoder (DL, hanya extractor)**: MLP 16-32-16-**8**-16-32-16, MSE+Adam,
  30 epoch; output cuma embedding → S3. Tidak pernah menentukan klaster.

## 5. Hasil (hafal tabel ini)

| Skenario | Sil↑ | DBI↓ | CH↑ |
|---|---|---|---|
| S1 KMeans scaled | 0,3132 | 1,3317 | 5.641 |
| **S2 KMeans+PCA** | **0,3493** | **1,1738** | **6.999** |
| S3 KMeans+AE | 0,3386 | 1,2439 | 6.143 |
| Agg-ward | 0,3128 | 1,3295 | 5.592 |
| DBSCAN | 0,0800 | 1,9652 | 172 |

Profil: Klaster 1 Loyal (tenure ±70 bln, spend ±716/bln, rate ±14,6) vs
Klaster 0 Promo-sensitive (tenure ±8 bln, spend ±71, promo 0,42).

## 6. Konsep yang wajib bisa dijelaskan
- **Silhouette** (−1..1, ↑baik): anggota dekat ke klasternya vs klaster tetangga.
- **Davies–Bouldin** (↓baik): rasio sebaran dalam vs jarak antar klaster.
- **Calinski–Harabasz** (↑baik): dispersi antar vs dalam klaster.
- **Elbow**: titik penurunan inertia (WCSS) melandai → kandidat k.
- **Dendrogram**: pohon merging; garis potong horizontal = jumlah klaster.
- **k-distance graph**: siku kurva = eps optimal DBSCAN.
- **StandardScaler**: z = (x−mean)/std; tidak mengubah bentuk distribusi.
- **Seed 42 + working_memory=256**: reproducibility; chunk kecil agar matriks
  jarak 37k muat RAM (hasil eksak sama).

## 7. Jawaban kilat tanya jawab
1. Kenapa k=2? → Silhouette max + data bimodal (bukan tebakan).
2. Di mana DL-nya? → Autoencoder → embedding S3; klaster tetap K-Means.
3. Kenapa DBSCAN rendah? → Beda fungsi (noise finder); data globular cocok K-Means.
4. Bukti reproducible? → Seed 42; run ulang di venv bersih metrik identik.
5. Demo? → Load preprocessor+pca8+kmeans_s2, predict 5 baris → `[0,0,0,0,1]`.
6. Limitasi? → k=2 sederhana; hierarchical subsample; tanpa ground truth;
   imputasi median; AE 30 epoch.
