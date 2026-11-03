# IF3170_G03_FiveAgents

## Kontrak bersama eksperimen Tugas Besar 2 IF3170

Lima model dikerjakan terpisah, tetapi hasilnya harus bisa dibandingkan. Aturan di bawah adalah kesepakatan kelompok, bukan tambahan dari spesifikasi. Keterangan kolom dan format submisi ada di [`data/README.md`](data/README.md).

### Data dan validasi

Jalankan notebook atau skrip dari akar repositori. Saat membaca kedua CSV, gunakan `dtype={"id": str}` agar nilai `id` tidak berubah. Kolom itu bukan fitur. Targetnya `is_canceled`, dengan nilai `0` untuk tidak batal dan `1` untuk batal.

Semua orang memakai indeks validasi yang sama, sekitar 20% dari tiap kelas dengan seed 42. Jangan ubah urutan atau jumlah baris `train.csv` sebelum menjalankan kode ini.

```python
import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv", dtype={"id": str})
y = train["is_canceled"].to_numpy()
assert set(np.unique(y)) == {0, 1}
rng = np.random.default_rng(42)
valid_idx = np.sort(np.concatenate([
    rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
    for label in (0, 1)
]))
train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
assert len(train_idx) + len(valid_idx) == len(train)
```

Latih dengan `train.iloc[train_idx]` dan nilai hasilnya pada `train.iloc[valid_idx]`. Catat jumlah baris serta jumlah contoh tiap kelas. Jangan gunakan `data/test.csv` untuk validasi atau gabungkan dengan data latih. Bila ada baris identik di kedua sisi split, bicarakan penanganannya sebelum membandingkan skor. Jika aturan split berubah, semua hasil lama perlu diberi penanda agar tidak tercampur dengan hasil baru.

### Pengerjaan tiap model

Di `src/experiments/<algoritma>/`, tiap orang boleh memakai `.py` atau `.ipynb`. Isinya mencakup pembacaan data, split, prapemrosesan, pelatihan, dan evaluasi. Pilihan cleaning, imputasi, encoding, scaling, seleksi fitur, penanganan ketidakseimbangan kelas, serta hiperparameter boleh berbeda. Tidak perlu membuat `preprocessing.py` bersama.

Pelajari parameter prapemrosesan hanya dari `train_idx`. Ini berlaku juga untuk imputasi, pemilihan fitur, dan penyeimbangan data. Terapkan transformasi yang sudah dipelajari pada validasi tanpa melatihnya ulang. Lakukan oversampling hanya pada bagian latih. Setelah memilih konfigurasi final, latih ulang prapemrosesan dan model pada seluruh `data/train.csv` sebelum memprediksi `test.csv`.

KNN, CART, Logistic Regression dengan L1/L2/Elastic Net, SVM dengan kernel linear/polynomial/Gaussian, serta Naive Bayes wajib diimplementasikan dari nol. NumPy boleh dipakai untuk perhitungan matematika. Pakai scikit-learn untuk pembanding dan metrik, bukan untuk menjalankan inti model dari nol. CART harus menangani fitur numerik, kategorikal, dan nilai null. Bila mengambil metode dari luar materi kuliah, tulis sumbernya di komentar kode dan daftar referensi laporan.

Setiap panggilan `predict` mengembalikan satu label integer `0` atau `1` untuk setiap baris, sesuai urutan input. Tuliskan bentuk fitur yang diharapkan model. Notebook final harus bisa dimulai dari CSV mentah tanpa lebih dulu menjalankan notebook eksperimen. Kode model dalam `.py` eksperimen boleh diimpor. Kalau implementasinya hanya ada di notebook eksperimen, sertakan kode yang diperlukan di notebook final.

### Membandingkan hasil

Ukur model dari nol dan pembanding scikit-learn pada `valid_idx` yang sama. Skor utama mengikuti Kaggle, yaitu `f1_score(y_valid, y_pred, labels=[0, 1], average="macro", zero_division=0)`. Sertakan confusion matrix dengan urutan `[0, 1]` dan precision, recall, serta F1 untuk masing-masing kelas. Hitung semua metrik ini dari prediksi validasi.

Untuk tiap percobaan, catat varian model, fitur dan prapemrosesan yang dipakai, hiperparameter, seed, skor validasi, metrik per kelas, serta skor pembanding. Grafik atau hasil run yang dipakai di laporan bisa disimpan di `output/` atau ditampilkan di notebook final.

Pilih satu konfigurasi dari nol dengan macro F1 tertinggi untuk tiap algoritma. Jika skornya seri, gunakan nama varian atau konfigurasi yang lebih dulu secara leksikografis. Pemenang antaralgoritma dipilih dengan cara yang sama. Jangan gunakan skor leaderboard Kaggle untuk menentukan pemenang atau mengubah split setelah melihat hasil test.

### Notebook final dan berkas hasil

Setiap algoritma mendapat satu notebook di `src/final-notebooks/<algoritma>/`. Notebook itu membaca CSV mentah, menunjukkan hasil validasi, melatih ulang konfigurasi terpilih pada seluruh data latih, lalu menyimpan model bersama prapemrosesannya di `output/final/model/<algoritma>/`. Nama dan format berkas bebas. Uji pemuatan pada sesi Python baru dan pastikan prediksinya identik. Jangan memuat berkas `pickle` atau `joblib` dari pihak yang tidak tepercaya.

Notebook submisi yang direncanakan berada di `src/final-notebooks/final.ipynb`. Alurnya dimulai dari CSV mentah, memperlihatkan alasan pemilihan pemenang, melatih ulang model tersebut, menyimpan dan memuatnya, lalu menghasilkan prediksi `data/test.csv` ke `output/submission.csv`. Berkas notebook ini belum dibuat.

CSV submisi hanya berisi kolom `id,is_canceled`, tanpa kolom indeks. Jumlah baris, urutan, dan nilai `id` harus sama persis dengan `data/test.csv`. Nilai prediksi hanya integer `0` atau `1`. Sebelum mengirim setidaknya satu submisi Kaggle, jalankan notebook dari awal pada sesi bersih. Periksa bahwa CSV yang dihasilkan sama dengan yang akan dikirim. Catat seed, konfigurasi, dan versi pustaka di notebook.

### Pembagian kerja

Arin mengerjakan KNN, Emilio Logistic Regression, Aufar DTL/CART, Kevin SVM, dan Vincent Naive Bayes. Setiap orang menyiapkan eksperimen, model pembanding, notebook final, berkas model, serta analisis untuk laporan.

Emilio menjaga agar split, seed, dan format hasil yang dipakai tetap sama. Ia juga mengintegrasikan notebook pemenang. Aufar memeriksa dukungan fitur numerik, kategorikal, dan null pada CART. Vincent memeriksa simpan/muat serta format submisi. Koordinasi ini tidak menggantikan pekerjaan pemilik model.

Jika ada usulan perubahan split, metrik, atau format keluaran, sepakati dulu bersama sebelum memakai skor baru untuk perbandingan. Pembagian tugas dan hasilnya perlu masuk ke laporan PDF saat finalisasi.
