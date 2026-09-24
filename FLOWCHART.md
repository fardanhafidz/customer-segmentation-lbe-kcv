# Flowchart Full — Notebook Customer Segmentation

> Cerminan 1:1 dari `customer_segmentation_unsupervised.ipynb` (33 cell).
> Setiap node mencantumkan nomor cell. Render diagram di GitHub / VS Code Markdown Preview.

## Diagram utama (start → finish)

```mermaid
flowchart TD
    C0([START<br/>Cell 0: Judul + aturan]) --> C1[Cell 1: Import + seed 42<br/>+ working_memory=256]
    C1 --> C3[Cell 3: Load data_train 37.5k<br/>+ data_test 12.5k]
    C3 --> EDA[Cell 5-10: EDA lengkap<br/>head - distribusi - korelasi - crosstab - bimodal]
    EDA --> C12[Cell 12: Preprocessing<br/>Imputasi + OneHot + StandardScaler<br/>fit TRAIN saja - 16 fitur]
    C12 --> C13[Cell 13: Bukti before-after scaling<br/>tabel + mean/std + histogram income]
    C13 --> PCA[Cell 15: PCA<br/>PCA-8 utk S2 + PCA-2 utk visual]
    C13 --> AE[Cell 16: DL Autoencoder 16-32-16-8<br/>30 epoch - output embedding 8-dim<br/>encoder.pt]
    PCA --> K[Cell 18: Elbow + Silhouette k=2..8<br/>pilih K_OPT = sil max]
    AE --> K
    C12 --> K
    K --> KOPT{K_OPT = 2?<br/>sil k=2 0.315 vs k=3 0.13}
    KOPT -->|Ya| H[Cell 19: Dendrogram ward<br/>subsample 2000]
    H --> DB[Cell 20: k-distance + sweep eps<br/>pilih EPS_OPT = 1.2]
    DB --> S1[Cell 22 S1: KMeans scaled<br/>fit train - predict test]
    DB --> S2[Cell 23 S2: KMeans PCA-8<br/>fit train - predict test]
    DB --> S3[Cell 24 S3: KMeans AE-emb<br/>fit train - predict test]
    DB --> AG[Cell 25 Lampiran: Agg ward<br/>fit_predict test]
    DB --> DN[Cell 25 Lampiran: DBSCAN<br/>fit_predict test]
    S1 --> EV[Cell 27: Evaluasi di TEST<br/>Silhouette - DBI - CH + bar chart]
    S2 --> EV
    S3 --> EV
    AG --> EV
    DN --> EV
    EV --> CHAMP{Juara S1-S3?<br/>sil tertinggi}
    CHAMP -->|S2 0.3493| VIZ[Cell 30: Scatter PCA-2<br/>4 panel + DBSCAN noise]
    VIZ --> PROF[Cell 31: Profiling centroid<br/>skala asli + label semantik otomatis]
    PROF --> DEMO[Cell 32: Simpan demo_input.csv<br/>load artefak - predict 5 baris<br/>TANPA re-fit]
    DEMO --> DEMOROUTE{Rute demo ikut juara}
    DEMOROUTE -->|AE| RAE[load encoder.pt + emb_scaler + kmeans_s3]
    DEMOROUTE -->|PCA| RPCA[load pca8 + kmeans_s2]
    DEMOROUTE -->|scaled| RS1[load kmeans_s1]
    RAE --> FIN([FINISH<br/>artefak + metrics.csv])
    RPCA --> FIN
    RS1 --> FIN
```

## Percabangan dan keputusan

| Titik cabang | Pilihan | Di kode | Hasil aktual |
|---|---|---|---|
| Representasi fitur | Scaled 16-dim / PCA-8 / AE-emb 8-dim | Cell 12/15/16 | 3 jalur paralel ke K-Means |
| K_OPT | k=2..8, silhouette max (train) | Cell 18 | k=2 (0,315 vs 0,13) |
| EPS_OPT | sweep 0,8–2,0, min_samples=10 | Cell 20 | eps=1,2 |
| Juara | silhouette tertinggi S1–S3 (test) | Cell 27 | S2 (0,3493) |
| Rute demo | cabang `if "AE" / elif "PCA" / else` | Cell 32 | jalur PCA (prediksi `[0,0,0,0,1]`) |
| DBSCAN noise | label −1 dikecualikan dari metrik | Cell 27 `scores()` | noise dilaporkan terpisah |

## Batas tegas ML vs DL (untuk slide)

- **DL (kuning): hanya Cell 16** — Autoencoder berhenti di `encoder.pt` / `emb_*.npy`.
- **ML klasik (biru): Cell 18–25** — K-Means / Agglomerative / DBSCAN penentu klaster.
- Tidak ada panah dari DL ke keputusan klaster selain sebagai input fitur S3.

## Pemetaan cell → file output

| Cell | Output |
|---|---|
| 12 | `artifacts/preprocessor.joblib` |
| 15 | `artifacts/pca8.joblib`, `pca2.joblib` |
| 16 | `artifacts/encoder.pt`, `autoencoder.pt`, `emb_train.npy`, `emb_test.npy`, `emb_scaler.joblib` |
| 22/23/24 | `artifacts/kmeans_s1/s2/s3.joblib` |
| 25 | `artifacts/dbscan.joblib` |
| 27 | `artifacts/metrics.csv` |
| 32 | `demo_input.csv` + prediksi demo |
