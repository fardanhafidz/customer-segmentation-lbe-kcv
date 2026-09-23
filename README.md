# Customer Segmentation — Unsupervised (K-Means / Hierarchical / DBSCAN + DL Autoencoder)

Tugas Final Project LBE KCV — bagian **Unsupervised**. Model penentu klaster akhir adalah
**ML klasik**; Deep Learning (MLP Autoencoder PyTorch) dipakai **hanya** sebagai
*feature extractor / preprocessing*.

## Isi repositori

| File | Keterangan |
|---|---|
| `customer_segmentation_unsupervised.ipynb` | Notebook utama (semua cell sudah di-run, output terlihat). Alur: A. Persiapan & Dataset → B. Parameter Optimal → C. 3 Skenario → D. Evaluasi → E. Visualisasi/Profiling/Demo |
| `data_train.csv` / `data_test.csv` | Data latih (37.500) / uji (12.500). Sumber: panitia LBE KCV |
| `requirements.txt` | Dependensi Python |
| `demo_input.csv` | 20 baris dari `data_test.csv` untuk demo (tinggal load model, tanpa training ulang) |
| `artifacts/` | Model pre-fitted: `preprocessor.joblib`, `pca8.joblib`, `pca2.joblib`, `encoder.pt`, `autoencoder.pt`, `emb_scaler.joblib`, `kmeans_s1/s2/s3.joblib`, `dbscan.joblib`, `metrics.csv`, `emb_*.npy` |

## Cara menjalankan

```bash
git clone https://github.com/fardanhafidz/customer-segmentation-lbe-kcv.git
cd customer-segmentation-lbe-kcv
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows; Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook customer_segmentation_unsupervised.ipynb
# atau tanpa browser:
python -m nbconvert --to notebook --execute customer_segmentation_unsupervised.ipynb --output customer_segmentation_unsupervised.ipynb --ExecutePreprocessor.timeout=1200
```

Notebook memakai `random_state=42` sehingga hasil konsisten saat di-run ulang.

## Arsitektur (flowchart)

```text
data_train.csv / data_test.csv
  → Imputasi median/modus + One-Hot + StandardScaler (fit di TRAIN saja)
  → 3 representasi: (S1) scaled 16-dim | (S2) PCA-8 | (S3) Autoencoder bottleneck 8-dim [DL, preprocessing saja]
  → K-Means (k=2, optimal via Elbow+Silhouette) sebagai penentu klaster
  → Lampiran pembanding: Agglomerative ward + DBSCAN (fit_predict di test)
  → Evaluasi di TEST: Silhouette↑, Davies-Bouldin↓, Calinski-Harabasz↑
```

## Hasil eksperimen (TEST, aktual dari notebook)

| Skenario | Silhouette↑ | DBI↓ | CH↑ |
|---|---|---|---|
| S1 KMeans scaled (k=2) | 0.3132 | 1.3317 | 5641.26 |
| **S2 KMeans + PCA-8 (k=2)** | **0.3493** | **1.1738** | **6998.81** |
| S3 KMeans + AE-embedding (k=2) | 0.3386 | 1.2439 | 6142.97 |
| Lampiran Agg-ward (k=2) | 0.3128 | 1.3295 | 5591.90 |
| Lampiran DBSCAN (eps=1.2, min_samples=10) | 0.0800 | 1.9652 | 171.68 |

**Kesimpulan:** juara S2; S3 (DL embedding) runner-up → ekstraktor DL valid;
DBSCAN berguna untuk outlier tetapi skor intrinsik lebih rendah.
Detail profiling semantik klaster ada di Bagian E notebook.

## Demo (tanpa training ulang)

Cell terakhir notebook melakukan: load `preprocessor` + model juara +
`demo_input.csv` → `predict` 5 baris pertama dan mencetak hasilnya.
Model Agglomerative/DBSCAN disimpan hanya sebagai artefak `fit_predict`
(keduanya tidak memiliki `.predict`).

## Keterbatasan

- K-Means asumsi klaster sferis; k=2 optimal secara silhouette (data terbelah dua rezim besar).
- Hierarchical O(n²) → dendrogram memakai subsample 2000 train.
- DBSCAN sensitif `eps`/`min_samples`; metrik dilaporkan pada titik non-noise.
- Missing diimputasi median/modus; Autoencoder dilatih ringan 30 epoch.