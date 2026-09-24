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

## Versi slide (disarankan — berorientasi PROSES, bukan kandidat)

```mermaid
flowchart LR
    A[1. Data<br/>37.5k train / 12.5k test] --> B[2. EDA<br/>distribusi - korelasi - missing]
    B --> C[3. Preprocessing<br/>imputasi + One-Hot + StandardScaler]
    C --> D[4. Representasi fitur<br/>S1 scaled - S2 PCA - S3 AE DL]
    D --> E[5. Clustering<br/>K-Means k=2 + pembanding]
    E --> F[6. Evaluasi<br/>Silhouette - DBI - CH]
    F --> G[7. Interpretasi + Demo<br/>2 segmen - pre-fitted]
```

Prinsipnya: S1/S2/S3 cukup disebut sebagai isi tahap 4 (satu baris), detail
perbandingannya hidup di **tabel hasil**, bukan di diagram. Pembanding
Hierarchical/DBSCAN disebut sekali di tahap 5. Dengan begini diagram ramping,
tidak terlihat "metodenya banyak banget", dan tidak ada yang disembunyikan.

## Versi alternatif (jika juri minta kandidat terlihat)

```mermaid
flowchart LR
    A[Data Customer<br/>37.5k train / 12.5k test] --> B[Preprocessing<br/>Imputasi + One-Hot + StandardScaler]
    B --> F[16 Fitur]
    F --> S1[S1<br/>Scaled 16D]
    F --> S2[S2<br/>PCA 8D]
    F --> S3[S3 - DL<br/>Autoencoder 8D<br/>hanya feature extractor]
    S1 --> K[K-Means<br/>k=2 dari Elbow+Silhouette]
    S2 --> K
    S3 --> K
    F -.-> P[Pembanding<br/>Hierarchical + DBSCAN]
    P -.-> E
    K --> E[Evaluasi<br/>Silhouette - DBI - CH]
    E --> J[Juara: S2 - 2 segmen<br/>+ demo pre-fitted]
```

```mermaid
flowchart LR
    A[Data Customer<br/>37.5k train / 12.5k test] --> B[Preprocessing<br/>Imputasi + One-Hot + StandardScaler]
    B --> F[16 Fitur]
    F --> S1[S1<br/>Scaled 16D]
    F --> S2[S2<br/>PCA 8D]
    F --> S3[S3 - DL<br/>Autoencoder 8D<br/>hanya feature extractor]
    S1 --> K[K-Means<br/>k=2 dari Elbow+Silhouette]
    S2 --> K
    S3 --> K
    F -.-> P[Pembanding<br/>Hierarchical + DBSCAN]
    P -.-> E
    K --> E[Evaluasi<br/>Silhouette - DBI - CH]
    E --> J[Juara: S2 - 2 segmen<br/>+ demo pre-fitted]
```

Catatan dari review draf: (1) kotak S3 wajib dilabel DL + beda warna (kuning) — di
drafmu DL-nya "tak terlihat" padahal S3 Autoencoder itulah komponen DL; (2) tambah
cabang putus-putus Hierarchical/DBSCAN sebagai pembanding agar tidak dikira
disembunyikan; (3) tambah sumber k=2 dan output akhir (2 segmen + demo).
