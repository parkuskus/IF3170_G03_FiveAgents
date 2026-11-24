# IF3170 G03 FiveAgents

![Sampul laporan akhir FiveAgents](docs/laporan-akhir/public/cover.png)

Repositori Tugas Besar 2 IF3170 Inteligensi Artifisial. Proyek ini membangun model klasifikasi from scratch untuk memprediksi pembatalan pemesanan akomodasi.

## Ringkasan

Lima keluarga algoritma dikembangkan dan dibandingkan dengan estimator scikit-learn.

- K-Nearest Neighbors
- Decision Tree Learning dengan ID3, C4.5, dan CART
- Logistic Regression
- Support Vector Machine
- Naive Bayes dan K-Dependence Bayesian Classifier

Notebook utama memuat analisis data, prapemrosesan, pelatihan baseline, evaluasi, analisis kesalahan, dan pembuatan berkas submission. Notebook per model serta modul eksperimen di `src/experiments/` memuat pengembangan dan evaluasi lanjutan.

## Dataset dan evaluasi

Dataset berada di `data/`. Berkas `train.csv` memuat target `is_canceled`, sedangkan `test.csv` tidak memuat target. Identitas pemesanan disimpan pada kolom `id` dan perlu dibaca sebagai teks. Keterangan dataset tersedia di [data/README.md](data/README.md).

Perbandingan baseline menggunakan pembagian validasi terstratifikasi dengan seed 42 dan metrik utama Macro F1. Confusion matrix serta precision, recall, dan F1 per kelas digunakan untuk melengkapi analisis. Data uji tidak digunakan untuk memilih konfigurasi model.

## Menjalankan proyek

Gunakan Python 3 dan pasang dependensi yang tercatat di `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

Jalankan notebook utama dari akar repositori

```bash
jupyter notebook notebooks/Notebook_FiveAgents_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb
```

Jalankan seluruh sel notebook secara berurutan. Notebook membaca data dari `data/` dan menghasilkan prediksi serta artefak di `output/`.

Notebook eksperimen per model tersedia di bawah ini

- [K-Nearest Neighbors](notebooks/KNN_Final-Pipeline.ipynb)
- [Decision Tree Learning](notebooks/DTL_Final-Pipeline.ipynb)
- [Logistic Regression](notebooks/Logistic_Regression_Final-Pipeline.ipynb)
- [Support Vector Machine](notebooks/SVM_Final-Pipeline.ipynb)
- [Naive Bayes](notebooks/Naive_Bayes_Final-Pipeline.ipynb)
- [Template pipeline model](notebooks/Model_Pipeline_Template.ipynb)

Jalankan pengujian dengan perintah berikut

```bash
python -m unittest discover -s tests -v
```

## Struktur repositori

- `data/` berisi dataset kompetisi dan keterangannya
- `notebooks/` berisi notebook utama, notebook per model, dan template eksperimen
- `src/eda/` berisi kode analisis eksploratif
- `src/experiments/` berisi implementasi model, prapemrosesan, runner eksperimen, serta pengujian khusus model
- `tests/` berisi pengujian baseline, pipeline akhir, dan pemilihan konfigurasi cost-sensitive
- `output/eda/` berisi visualisasi analisis data
- `output/experiments/` dan `output/knn/` berisi hasil evaluasi eksperimen
- `output/final/model/` menjadi lokasi artefak model akhir
- `docs/laporan-akhir/` berisi sumber dan PDF laporan akhir
- `docs/` juga memuat kebijakan AI dan surat pernyataan AI
- `extra/` berisi spesifikasi tugas dan dokumen arsitektur serta pembagian kerja

## Artefak utama

Berkas submission utama berada di `output/submission.csv`. Sejumlah notebook eksperimen juga menghasilkan submission khusus model. Visualisasi pohon DTL dan graf Bayesian Network tersedia di `output/`, sementara hasil eksperimen tersimpan di folder keluaran masing-masing.

Laporan akhir dapat dibaca di [IF3170_G11_ThreeAgents.pdf](docs/laporan-akhir/IF3170_G11_ThreeAgents.pdf). Sumber LaTeX dan daftar referensinya berada di folder yang sama.
