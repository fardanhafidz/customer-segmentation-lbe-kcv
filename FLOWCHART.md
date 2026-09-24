# Flowchart — Pengelompokan Pelanggan (Garis Besar)

> Versi ringkas selevel fase. Detail per cell ada di git history (`FLOWCHART.md` versi detail).

```mermaid
flowchart TD
    A[Data: data_train 37.5k + data_test 12.5k] --> B[EDA: distribusi - korelasi - missing]
    B --> C[Preprocessing: imputasi + One-Hot + StandardScaler]
    C --> D1[S1: fitur scaled]
    C --> D2[S2: fitur PCA-8]
    C --> DL[DL: Autoencoder - hanya ambil embedding]
    DL --> D3[S3: fitur embedding]
    D1 --> K[K-Means k=2 - penentu klaster]
    D2 --> K
    D3 --> K
    C --> P[Pembanding: Hierarchical + DBSCAN]
    K --> E[Evaluasi di data_test: Silhouette - DBI - CH]
    P --> E
    E --> F[Juara: S2 - profiling 2 segmen - demo pre-fitted]
```

## Keputusan kunci (3 saja)

| Keputusan | Hasil |
|---|---|
| k optimal (Elbow + Silhouette) | k = 2 |
| eps DBSCAN (k-distance) | 1,2 |
| Juara (silhouette tertinggi) | S2 K-Means + PCA-8 (0,3493) |

## Batas ML vs DL

- **DL**: Autoencoder berhenti di embedding — tidak menentukan klaster.
- **ML klasik**: K-Means (utama) + Hierarchical + DBSCAN (pembanding).
