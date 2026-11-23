# Decision Tree Learning (DTL)

Eksperimen ID3, C4.5, dan CART dari nol untuk memprediksi `is_canceled`. Dokumen ini membahas runner DTL di `src/experiments/decision_tree/`, bukan baseline yang dipilih secara terpisah oleh notebook utama.

## Overview

Ketiga varian DTL dievaluasi pada validation split bersama dengan seed 42 dan Macro F1. Setiap varian scratch dibandingkan dengan `DecisionTreeClassifier` dari `scikit-learn`. Eksperimen CART tambahan memilih konfigurasi pertumbuhan, *cost-complexity post-pruning*, dan threshold melalui 5-fold stratified CV pada bagian latih. Holdout validation tidak dipakai untuk memilih parameter.

Hasil tersimpan menunjukkan **DTL-v08 (CART post-pruning)** sebagai pipeline final eksperimen ini, dengan Macro F1 holdout **0,8371**. Hasil tersebut tidak otomatis menggantikan baseline terpilih di notebook utama.

## Program Features

- **Tiga varian dari nol.** ID3 (*information gain*), C4.5 (*gain ratio*), dan CART (Gini) mendukung fitur numerik, kategorikal, dan nilai kosong.
- **Evaluasi dan pembanding.** Macro F1, metrik per kelas, confusion matrix, serta waktu eksekusi dihitung pada split yang sama. Model pembanding menggunakan `scikit-learn`.
- **Tuning CART.** Konfigurasi pertumbuhan, `ccp_alpha`, dan threshold dipilih lewat CV dengan preprocessing yang di-*fit* ulang pada setiap fold.
- **Submission berversi.** Setiap varian scratch di-*refit* pada seluruh `train.csv`. Submission terbaru dan snapshot bernomor disimpan per varian.

## Dependencies & Prerequisites

- Python 3 dengan NumPy, pandas, dan scikit-learn.
- `data/train.csv`, `data/test.csv`, dan `data/sample_submission.csv` di akar repositori.

Pasang dependensi dari akar repositori.

```bash
python -m pip install numpy pandas scikit-learn
```

Format data dan aturan split bersama dijelaskan di [`data/README.md`](../../../data/README.md) dan [`README.md`](../../../README.md).

## How to Run the Program

Jalankan perintah dari akar repositori.

**Seluruh varian**

```bash
python src/experiments/decision_tree/run_all.py
```

**Satu varian**

```bash
python src/experiments/decision_tree/ID3/run.py
python src/experiments/decision_tree/C4.5/run.py
python src/experiments/decision_tree/CART/run.py
```

Jalankan hanya perintah untuk varian yang diperlukan. Runner CART menambahkan tuning 5-fold sehingga membutuhkan waktu lebih lama daripada baseline. Hasil seluruh varian ditulis ke `output/experiments/decision_tree/validation.csv` dan `validation.md`. Hasil runner tunggal disimpan di subfolder varian. Kandidat CART tercatat di `CART/pruning.csv` dan `CART/pruning.md` setelah runner CART dijalankan.

**Eksperimen class weighting (terpisah, tanpa submission)**

```bash
python src/experiments/decision_tree/run_class_weighted.py
```

Hasilnya disimpan di `output/experiments/decision_tree/CART_class_weight/`.

**Ablasi diagnostik CART (tanpa submission)**

```bash
python -m src.experiments.decision_tree.ablation
```

`output/experiments/decision_tree/ablation_holdout.csv` mencatat tiga konfigurasi pertumbuhan dari runner dan enam variasi satu parameter relatif terhadap konfigurasi final. Setiap konfigurasi dievaluasi pada empat nilai alpha dan tiga ambang prediksi, dengan jumlah daun dan confusion matrix pada holdout yang sama. Preprocessing di-*fit* pada bagian latih saja. Konfigurasi final tetap dipilih melalui 5-fold CV pada bagian latih oleh pipeline final. Jangan memilih parameter baru berdasarkan skor holdout diagnostik ini.

**Analisis kesalahan model final (tanpa submission)**

```bash
python -m src.experiments.decision_tree.error_slices
```

Runner melatih ulang DTL-v08 hanya pada bagian latih, lalu menyimpan jumlah dan tingkat kesalahan per kelompok pada `output/experiments/decision_tree/error_slices.csv`. Kelompok `days_before_arrival` ialah 0–30, 31–147, 148 ke atas, dan kosong. Tingkat luput menggunakan seluruh kelas positif dalam kelompok sebagai penyebut, sedangkan tingkat alarm palsu menggunakan seluruh kelas negatif. `holdout` hanya dipakai untuk pelaporan kesalahan, bukan memilih parameter.

**Tes DTL**

```bash
python -m unittest src.experiments.decision_tree.test_baselines -v
```

## Struktur Direktori

```text
src/experiments/decision_tree/
├── README.md
├── shared.py
├── run_all.py
├── run_class_weighted.py
├── test_baselines.py
├── ID3/run.py
├── C4.5/run.py
└── CART/run.py

output/experiments/decision_tree/
├── validation.csv
├── validation.md
├── ID3/
├── C4.5/
├── CART/
│   ├── pruning.csv
│   ├── pruning.md
│   ├── submission.csv
│   ├── submission.json
│   └── versions/
└── CART_class_weight/
```

Berkas output dibuat saat runner dijalankan. Berkas metadata JSON untuk submission baru tidak otomatis tersedia pada snapshot lama.

## Panduan Pengembangan

### Dataset & validation split

Runner membaca `id` sebagai teks, mengecualikan `id` dan `is_canceled` dari fitur, serta memakai split stratified sekitar 20% per kelas dengan seed 42. Statistik preprocessing dipelajari dari bagian latih saja. `data/test.csv` hanya digunakan untuk membuat submission, bukan untuk memilih model atau menghitung skor.

### Model implementation & preprocessing

Semua varian scratch memakai imputasi median untuk fitur numerik. Kategori kosong menjadi `__MISSING__`, sedangkan kategori yang tidak terlihat saat *fit* menjadi `__UNKNOWN__`. Pembanding `scikit-learn` memakai one-hot encoding karena membutuhkan input numerik.

ID3 membagi kategori menjadi beberapa cabang dengan *information gain*. C4.5 memakai *gain ratio* setelah menyaring kandidat berdasarkan *information gain*. CART memakai split biner dengan Gini. Fitur numerik memakai threshold biner. Ketiga baseline scratch dibatasi `max_depth=8`, `min_samples_split=80`, dan `min_samples_leaf=40`. Pohon CART sebelum pruning dapat tumbuh tanpa batas kedalaman, dengan `min_samples_split=2` dan `min_samples_leaf=1`. Kandidat threshold numerik dibatasi paling banyak 64 titik, dan kategori CART dengan lebih dari 10 nilai diuji secara *one-versus-rest*.

Pembanding memakai `DecisionTreeClassifier` dengan entropy untuk ID3, entropy dan `ccp_alpha=0.001` untuk C4.5, serta Gini untuk CART. Ketiganya memakai batas kedalaman dan ukuran daun baseline yang sama. Pembanding C4.5 bukan implementasi *gain ratio* yang identik.

### Evaluation & comparison

Hasil validasi memuat Macro F1, precision, recall, F1 per kelas, confusion matrix, waktu eksekusi, serta konfigurasi model. `run_all.py` memberi ID dan peran eksperimen berikut pada hasil yang baru dihasilkan.

| ID | Percobaan | Peran |
| --- | --- | --- |
| DTL-v01–v02 | ID3 scratch / `scikit-learn` | baseline / pembanding |
| DTL-v03–v04 | C4.5 scratch / `scikit-learn` | baseline / pembanding |
| DTL-v05–v06 | CART scratch / `scikit-learn` | baseline / pembanding |
| DTL-v07 | CART dengan konfigurasi pertumbuhan terpilih, tanpa pruning | eksperimen |
| **DTL-v08** | **CART dengan konfigurasi terpilih, post-pruning, dan threshold hasil CV** | **FINAL** |
| DTL-v09 | CART dengan *class weighting* (runner terpisah) | eksperimen |

Eksperimen *modern preprocessing* di [`notebooks/decision_tree_end_to_end.ipynb`](../../../notebooks/decision_tree_end_to_end.ipynb) disimpan terpisah dan tidak mengganti pipeline final.

### Final pipeline & submission

Untuk **DTL-v08**, runner memilih konfigurasi pertumbuhan, `ccp_alpha`, dan threshold berdasarkan Macro F1 pada 5-fold CV bagian latih. Pada hasil tersimpan, konfigurasi terpilih adalah `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `ccp_alpha=0.0001`, dan threshold `0.62`. Setelah evaluasi holdout, runner melatih ulang preprocessing dan CART pada seluruh `train.csv`, menerapkan pruning dan threshold tersebut, lalu menulis `CART/submission.csv`.

Setiap eksekusi berikutnya menulis `versions/submission_vNNN.csv` beserta `submission_vNNN.json` yang mencatat ID eksperimen, peran, model, dan konfigurasi. `submission.csv` dan `submission.json` di folder varian menunjuk keluaran terbaru. Angka `vNNN` menunjukkan urutan submission, **bukan** ID eksperimen `DTL-vNN`. Snapshot lama tidak diubah atau diberi metadata yang tidak dapat diverifikasi.
