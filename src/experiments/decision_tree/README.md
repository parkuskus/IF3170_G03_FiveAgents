# Baseline Decision Tree

Jalankan dari akar repositori:

```powershell
python src/experiments/decision_tree/run_all.py
```

Perintah ini melatih tiga implementasi dari nol dan tiga pembanding scikit-learn pada split validasi kontrak README. Ringkasan macro F1, metrik per kelas, confusion matrix, dan waktu run ditulis ke `output/experiments/decision_tree/`. Setiap implementasi scratch juga dilatih ulang dengan seluruh `train.csv` untuk membuat `submission.csv` terbaru di subfolder `ID3/`, `C4.5/`, dan `CART/`, mengikuti ID dan kolom dari `sample_submission.csv`. Snapshot versi disimpan berurutan di subfolder `versions/` pada setiap folder algoritma.

Untuk menjalankan satu varian, gunakan `ID3/run.py`, `C4.5/run.py`, atau `CART/run.py`. Validasi dan `submission.csv` varian itu disimpan di subfolder nama algoritma pada lokasi output yang sama. Runner CART menumbuhkan pohon maksimal (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`), lalu memilih cost-complexity pruning alpha dan threshold dari kandidat dengan 5-fold stratified CV pada bagian latih. Preprocessing di-fit ulang untuk tiap fold; holdout validation tidak dipakai untuk seleksi. Ringkasan kandidat pruning ada di `output/experiments/decision_tree/CART/pruning.csv` dan `pruning.md`.

## Preprocessing bersama

Semua model menerima hasil transformasi yang sama. Fitur numerik diimputasi dengan median data latih. Nilai kategori kosong menjadi `__MISSING__`; kategori yang tidak ditemukan saat fit menjadi `__UNKNOWN__`. Statistik preprocessing hanya dipelajari dari bagian latih. `id` dan target tidak dipakai sebagai fitur. Pembanding scikit-learn memakai one-hot encoding atas hasil preprocessing yang sama karena estimator memerlukan fitur numerik.

## Variasi dari nol

- ID3 memilih information gain. Fitur kategori membuat cabang per nilai. Fitur numerik memakai threshold biner.
- C4.5 memilih gain ratio dengan perlakuan fitur sama seperti di atas.
- CART memilih penurunan Gini dan split kategori biner.

Ketiga baseline memakai kedalaman maksimum 8, paling sedikit 80 baris untuk split, dan paling sedikit 40 baris per daun. Eksperimen CART maksimal tidak menerapkan batas kedalaman, jumlah minimum split 2 dan daun 1 sebelum pruning. Kandidat threshold numerik tetap dibatasi paling banyak 64 titik yang tersebar pada nilai terurut; kategori CART dengan lebih dari 10 nilai memakai kandidat one-versus-rest. Batas pencarian split ini menjaga runtime dan berarti pohon maksimal relatif terhadap kandidat yang dicoba.

Pembanding scikit-learn memakai `DecisionTreeClassifier`. ID3 menggunakan entropy, C4.5 menggunakan entropy dan `ccp_alpha=0.001`, sedangkan CART menggunakan Gini. Semua memakai batas kedalaman dan ukuran daun yang sama.
