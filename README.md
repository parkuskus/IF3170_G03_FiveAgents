# IF3170_G03_FiveAgents

Implementasi Tugas Besar 2 IF3170 Inteligensi Artifisial: membangun classification model from scratch untuk memprediksi apakah pemesanan akomodasi dibatalkan.

## Overview

Proyek ini menganalisis dataset pemesanan hotel dan memprediksi cancellation status (`is_canceled`). Lima keluarga algoritma—K-Nearest Neighbors (KNN), Decision Tree Learning (DTL), Logistic Regression, Support Vector Machine (SVM), dan Naive Bayes—diimplementasikan from scratch, lalu dibandingkan dengan implementasi `scikit-learn`. Evaluasi memakai Macro F1 pada validation split bersama.

Notebook utama mencakup data understanding, preprocessing, training dan evaluasi baseline, error analysis, serta pembuatan submission. Template per model disediakan untuk eksperimen improvement.

## Program Features

- **Data analysis** dengan chart target distribution, missingness, feature distribution per class, cancellation rate per kategori, dan train–test feature drift.
- **Lima algoritma from scratch** dengan varian baseline: KNN; ID3, C4.5, dan CART; Logistic Regression; SVM; serta Naive Bayes.
- **Evaluasi konsisten** menggunakan validation split dengan seed 42, Macro F1, confusion matrix, serta precision, recall, dan F1 tiap class.
- **scikit-learn baseline** sebagai pembanding implementasi sendiri.
- **Submission pipeline** yang melatih ulang baseline terpilih pada seluruh training set, menyimpan model, dan membuat prediksi untuk test set.

## Dependencies & Prerequisites

- Python 3.
- Jupyter Notebook atau JupyterLab untuk menjalankan notebook.
- Paket Python: NumPy, pandas, scikit-learn, Matplotlib, dan NetworkX.

Pasang dependensi dengan:

```bash
python -m pip install numpy pandas scikit-learn matplotlib networkx jupyter
```

Competition dataset tersedia di `data/`. Keterangan kolom, format submission, dan sumber data dapat dilihat pada [`data/README.md`](data/README.md).

## How to Run the Program

Jalankan perintah dari akar repositori.

**Notebook utama**

```bash
jupyter notebook notebooks/Notebook_FiveAgents_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
```

Jalankan seluruh sel dari awal. Notebook membaca `data/train.csv` dan `data/test.csv`, membandingkan baseline, menyimpan model terbaik beserta preprocessing-nya di `output/final/model/<algoritma>/`, dan membuat `output/submission.csv`.

**Template eksperimen per model**

Pilih notebook yang sesuai, lalu salin dan modifikasi untuk eksperimen improvement:

| Model | Notebook |
| --- | --- |
| KNN | [`KNN_Final-Pipeline.ipynb`](notebooks/KNN_Final-Pipeline.ipynb) |
| Decision Tree Learning (ID3, C4.5, CART) | [`DTL_Final-Pipeline.ipynb`](notebooks/DTL_Final-Pipeline.ipynb) |
| Logistic Regression | [`Logistic_Regression_Final-Pipeline.ipynb`](notebooks/Logistic_Regression_Final-Pipeline.ipynb) |
| SVM | [`SVM_Final-Pipeline.ipynb`](notebooks/SVM_Final-Pipeline.ipynb) |
| Naive Bayes | [`Naive_Bayes_Final-Pipeline.ipynb`](notebooks/Naive_Bayes_Final-Pipeline.ipynb) |

`Model_Pipeline_Template.ipynb` adalah sumber template generik. Template khusus belum berisi implementasi improvement; lengkapi bagian yang ditandai sebelum menjalankannya.

**Tes**

```bash
python -m unittest discover -s tests -v
```

## Struktur Direktori

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── README.md
├── notebooks/
│   ├── Notebook_FiveAgents_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
│   ├── Model_Pipeline_Template.ipynb
│   └── *_Final-Pipeline.ipynb
├── src/
│   ├── eda/
│   ├── experiments/
│   └── final-notebooks/
├── tests/
│   ├── test_baseline_models.py
│   └── test_baseline_notebook.py
├── docs/
│   └── laporan-akhir/
├── output/
│   ├── final/model/
│   └── submission.csv
└── README.md
```

## Panduan Pengembangan

### Dataset & validation split

Saat membaca CSV, gunakan `dtype={"id": str}` agar nilai ID tidak berubah. Kolom `id` bukan feature. Target `is_canceled` bernilai `0` untuk pemesanan yang tidak dibatalkan dan `1` untuk pemesanan yang dibatalkan.

Semua eksperimen memakai validation indices yang sama: sekitar 20% dari setiap class dengan seed 42. Jangan mengubah urutan atau jumlah baris `train.csv` sebelum membuat split.

```python
import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv", dtype={"id": str})
y = train["is_canceled"].to_numpy()
assert set(np.unique(y)) == {0, 1}
rng = np.random.default_rng(42)
valid_idx = np.sort(np.concatenate([
    rng.permutation(np.flatnonzero(y == label))[
        :int(np.ceil(0.2 * np.sum(y == label)))
    ]
    for label in (0, 1)
]))
train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
assert len(train_idx) + len(valid_idx) == len(train)
```

Fit model menggunakan `train.iloc[train_idx]` dan evaluasi pada `train.iloc[valid_idx]`. Catat jumlah baris serta jumlah contoh tiap class. Jangan gunakan `data/test.csv` untuk validation atau menggabungkannya dengan training set. Jika ada baris identik di kedua bagian split, sepakati penanganannya sebelum membandingkan skor. Perubahan aturan split harus ditandai agar hasil lama dan baru tidak tercampur.

### Model implementation & preprocessing

Implementasi baseline from scratch berada pada sel algoritma masing-masing di notebook utama. Eksperimen improvement dikerjakan menggunakan template per model; kode tambahan dapat disimpan sebagai `.py` atau `.ipynb` di `src/experiments/<algoritma>/`.

KNN, ID3, C4.5, CART, Logistic Regression dengan L1/L2/Elastic Net, SVM dengan linear/polynomial/Gaussian kernel, serta Naive Bayes wajib diimplementasikan from scratch. NumPy boleh digunakan untuk perhitungan matematika. Gunakan scikit-learn untuk baseline comparison, transformation, dan metrics—bukan sebagai inti implementasi. Decision tree harus menangani numeric dan categorical features, missing values, serta unseen categories. Cantumkan sumber di komentar kode dan daftar referensi laporan jika menggunakan metode di luar materi kuliah.

Pilihan data cleaning, imputation, encoding, scaling, feature selection, class imbalance handling, dan hyperparameter dapat berbeda antar eksperimen. Preprocessing parameters hanya boleh dipelajari dari `train_idx`; terapkan hasilnya pada validation set tanpa melakukan refit. Oversampling hanya dilakukan pada training set. Setelah final configuration dipilih, fit ulang preprocessing dan model menggunakan seluruh `data/train.csv` sebelum menghasilkan predictions untuk `test.csv`.

Setiap pemanggilan `predict` harus menghasilkan satu integer label `0` atau `1` untuk setiap baris, mengikuti urutan input. Dokumentasikan feature shape yang diharapkan model. Final notebook harus dapat dijalankan dari raw CSV tanpa menjalankan notebook eksperimen terlebih dahulu. Kode model dari eksperimen `.py` boleh diimpor; jika implementasi hanya tersedia di notebook eksperimen, sertakan kode yang diperlukan pada final notebook.

### Evaluation & comparison

Evaluasi from-scratch implementation dan scikit-learn baseline pada `valid_idx` yang sama. Main metric mengikuti kompetisi Kaggle:

```python
f1_score(y_valid, y_pred, labels=[0, 1], average="macro", zero_division=0)
```

Sertakan confusion matrix dengan urutan class `[0, 1]`, serta precision, recall, dan F1 untuk masing-masing class. Semua metrics dihitung dari validation predictions.

Untuk setiap eksperimen, catat model variant, features dan preprocessing, hyperparameters, seed, validation score, metrics per class, serta baseline score. Simpan chart atau run results yang digunakan dalam laporan di `output/` atau tampilkan pada final notebook.

Notebook utama memilih from-scratch baseline dengan Macro F1 tertinggi di antara semua varian; jika skornya sama, nama varian menjadi pemecah seri secara leksikografis. Hasil improvement dievaluasi pada validation split yang sama, tetapi tidak otomatis menggantikan baseline. Jangan memakai skor leaderboard Kaggle untuk memilih pemenang atau mengubah split setelah melihat test set.

### Final notebook & submission

Notebook utama membaca raw CSV, menampilkan hasil baseline, fit ulang pemenang pada seluruh training set, lalu menyimpan model bersama preprocessing-nya di `output/final/model/<algoritma>/`. Uji load model pada sesi Python baru dan pastikan prediksinya identik. Jangan memuat berkas `pickle` atau `joblib` dari sumber yang tidak tepercaya.

Berkas `output/submission.csv` hanya memiliki kolom `id,is_canceled`, tanpa index column. Jumlah baris, urutan, dan nilai `id` harus sama persis dengan `data/test.csv`; predicted label hanya integer `0` atau `1`. Sebelum mengirim submission ke Kaggle, jalankan notebook dari awal pada clean session dan pastikan CSV yang dihasilkan sama dengan berkas yang akan dikirim. Catat seed, configuration, dan package versions di notebook.

### Pembagian kerja

Baseline kelima algoritma diintegrasikan di notebook utama oleh Aufar. Arin mengerjakan improvement KNN, Emilio Logistic Regression, Aufar DTL (ID3, C4.5, dan CART), Kevin SVM, dan Vincent Naive Bayes. Setiap anggota menyiapkan eksperimen improvement, pembanding, serta analisis laporan pada template model masing-masing.

Semua eksperimen improvement menggunakan validation split, seed, dan output format yang sama dengan baseline. Aufar memeriksa dukungan numeric dan categorical features, missing values, serta unseen categories pada DTL. Pemilik model memeriksa model persistence/reloading dan submission format notebook improvement. Usulan perubahan split, metric, atau output format harus disepakati sebelum scores baru dipakai untuk perbandingan. Pembagian tugas dan hasil eksperimen dicantumkan dalam laporan PDF saat finalisasi.
