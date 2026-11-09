# Ringkasan Eksperimen SVM & Komparasi `scikit-learn` (Tahap 3 — Target > 0.85)

## 1. Konfigurasi Split dan Prapemrosesan
- **Split Kontrak (`seed=42`, Stratified 80:20):**
  - `train_idx`: **21.898 sampel** (Kelas `0`: 14.723, Kelas `1`: 7.175)
  - `valid_idx`: **5.475 sampel** (Kelas `0`: 3.681, Kelas `1`: 1.794)
- **Prapemrosesan Lanjutan (`SVMPreprocessor`, fit hanya pada `train_idx`):**
  - Jumlah fitur akhir: **90 fitur**
  - Indikator missing values (7 fitur flag).
  - Imputasi median pada numerik dan modus pada biner/kategori.
  - Rekayasa fitur domain perhotelan:
    - Non-linear binning lead time (`days_before_arrival`) dari sangat singkat (<= 3 hari) hingga sangat lama (> 210 hari).
    - Pola khusus jumlah *special requests* (`sr_0`, `sr_1`, `sr_2`, `sr_ge3`).
    - Atribut tamu & durasi (`rate_per_guest_night`, `is_zero_nights`, `is_long_stay`, `weekend_ratio`, `is_single`, `is_couple`, `is_family`).
    - Rasio tarif terhadap rata-rata kategori kamar dan kanal pemesanan.
    - Sinyal pembatalan masa lalu & tamu berulang (`is_clean_returning`).
    - Interaksi kunci non-linear (OTA dengan lead time tinggi, OTA tanpa special request, pemesanan korporat, interaksi tarif & waktu).
    - Fitur siklikal & tren waktu (`days_since_start`, `dayofweek`, `dayofyear`, `month_sin`, `month_cos`, `is_month_end`).
    - *Out-of-fold target encoding* dengan smoothing untuk kombinasi segmen (`booking_channel`, `meal_package`, `room_category`, `ym`, `lead_ch`, `lead_sr`, `ch_room_meal`).
    - Estimator probabilitas interaksi non-linear berbasis pohon (`tree_prob`) yang dilatih strictly OOF pada train.
  - Standardisasi z-score dengan `StandardScaler`.

---

## 2. Hasil Eksperimen 18 Kandidat `SVMFromScratch` (Diurutkan berdasarkan Macro F1 Validasi)

| Rank | Variant | Kernel | $C$ | Degree | Gamma | Coef0 | Class Weight | Default Macro F1 | Best Tuned Macro F1 | Best Threshold | F1 (Kelas 0) | F1 (Kelas 1) | Fit Time (s) | Support Vectors |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`rbf_c1.0_scale_none`** | **rbf** | **1.0** | **3** | **scale** | **0.0** | **None** | **0.8544** | **0.8573** | **-0.160** | **0.9074** | **0.8072** | **10.77** | **7464** |
| 2 | `rbf_c1.0_scale_bal` | rbf | 1.0 | 3 | scale | 0.0 | balanced | 0.8532 | 0.8570 | +0.210 | 0.9071 | 0.8068 | 10.80 | 8063 |
| **3** | **`linear_c0.5_none`** | **linear** | **0.5** | **1** | **scale** | **0.0** | **None** | **0.8565** | **0.8568** | **-0.030** | **0.9084** | **0.8053** | **2.47** | **10697** |
| 4 | `linear_c1.0_none` | linear | 1.0 | 1 | scale | 0.0 | None | 0.8565 | 0.8568 | -0.030 | 0.9084 | 0.8053 | 2.64 | 10697 |
| 5 | `linear_c1.0_bal` | linear | 1.0 | 1 | scale | 0.0 | balanced | 0.8565 | 0.8568 | -0.030 | 0.9084 | 0.8053 | 3.34 | 10697 |
| 6 | `linear_c2.0_none` | linear | 2.0 | 1 | scale | 0.0 | None | 0.8565 | 0.8568 | -0.030 | 0.9084 | 0.8053 | 2.61 | 10697 |
| 7 | `linear_c2.0_bal` | linear | 2.0 | 1 | scale | 0.0 | balanced | 0.8565 | 0.8568 | -0.030 | 0.9084 | 0.8053 | 2.14 | 10697 |
| 8 | `linear_c4.0_bal` | linear | 4.0 | 1 | scale | 0.0 | balanced | 0.8565 | 0.8568 | -0.030 | 0.9084 | 0.8053 | 3.44 | 10697 |
| **9** | **`poly_d2_c2.0_r1_bal`** | **poly** | **2.0** | **2** | **scale** | **1.0** | **balanced** | **0.8502** | **0.8557** | **+0.290** | **0.9100** | **0.8014** | **12.92** | **8753** |
| 10 | `rbf_c2.0_scale_bal` | rbf | 2.0 | 3 | scale | 0.0 | balanced | 0.8524 | 0.8550 | +0.310 | 0.9067 | 0.8033 | 8.56 | 8344 |
| 11 | `poly_d3_c1.0_r1_bal` | poly | 1.0 | 3 | scale | 1.0 | balanced | 0.8508 | 0.8548 | +0.400 | 0.9086 | 0.8010 | 9.80 | 8889 |
| 12 | `rbf_c3.0_scale_bal` | rbf | 3.0 | 3 | scale | 0.0 | balanced | 0.8504 | 0.8548 | +0.210 | 0.9077 | 0.8018 | 10.81 | 8390 |
| 13 | `rbf_c4.0_scale_bal` | rbf | 4.0 | 3 | scale | 0.0 | balanced | 0.8488 | 0.8538 | +0.200 | 0.9066 | 0.8010 | 9.62 | 8553 |
| 14 | `poly_d3_c1.0_r1_none` | poly | 1.0 | 3 | scale | 1.0 | None | 0.8511 | 0.8532 | +0.150 | 0.9090 | 0.7974 | 11.51 | 8579 |
| 15 | `poly_d3_c4.0_r1_bal` | poly | 4.0 | 3 | scale | 1.0 | balanced | 0.8532 | 0.8531 | -0.010 | 0.9056 | 0.8006 | 9.19 | 10066 |
| 16 | `poly_d3_c2.0_r1_bal` | poly | 2.0 | 3 | scale | 1.0 | balanced | 0.8524 | 0.8530 | +0.020 | 0.9061 | 0.8000 | 10.05 | 9899 |
| 17 | `rbf_c4.0_g0.01_bal` | rbf | 4.0 | 3 | 0.01 | 0.0 | balanced | 0.8489 | 0.8529 | +0.330 | 0.9063 | 0.7995 | 8.91 | 8456 |
| 18 | `poly_d3_c3.0_r1_bal` | poly | 3.0 | 3 | scale | 1.0 | balanced | 0.8530 | 0.8529 | -0.010 | 0.9054 | 0.8003 | 10.74 | 10053 |

**Pemenang Keseluruhan SVM Dari Nol:**
- **`rbf_c1.0_scale_none`** (`kernel="rbf"`, `C=1.0`, `gamma="scale"`) dengan **Macro F1 = 0.8573** (F1 Kelas 0: `0.9074`, F1 Kelas 1: `0.8072`).
- Seluruh 3 kernel dari nol berhasil melampaui batas ambang **0.8500**:
  - **RBF**: **0.8573**
  - **Linear**: **0.8568**
  - **Polynomial**: **0.8557**

---

## 3. Komparasi Head-to-Head Terbaik Tiap Kernel dengan `sklearn.svm.SVC`

| Kernel | Variant | Implementasi | Macro F1 | Prec 0 | Rec 0 | F1 0 | Prec 1 | Rec 1 | F1 1 | Confusion Matrix `[[TN, FP], [FN, TP]]` | Slack Valid | Fit Time (s) | Predict Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF** | `rbf_c1.0_scale_none` | `SVMFromScratch` | **0.8573** | 0.9031 | 0.9117 | 0.9074 | 0.8152 | 0.7993 | 0.8072 | `[[3356, 325], [360, 1434]]` | 1594.02 | **10.77** | **0.25** |
| **RBF** | `rbf_c1.0_scale_none` | `sklearn.svm.SVC` | **0.8575** | 0.8994 | 0.9180 | 0.9086 | 0.8242 | 0.7893 | 0.8064 | `[[3379, 302], [378, 1416]]` | 1583.91 | 25.84 | 5.64 |
| **Linear** | `linear_c0.5_none` | `SVMFromScratch` | **0.8568** | 0.8981 | 0.9190 | 0.9084 | 0.8255 | 0.7860 | 0.8053 | `[[3383, 298], [384, 1410]]` | 1783.72 | **2.47** | **0.001** |
| **Linear** | `linear_c0.5_none` | `sklearn.svm.SVC` | **0.8569** | 0.9039 | 0.9098 | 0.9069 | 0.8124 | 0.8016 | 0.8070 | `[[3349, 332], [356, 1438]]` | 1682.81 | 29.83 | 0.94 |
| **Polynomial** | `poly_d2_c2.0_r1_bal` | `SVMFromScratch` | **0.8557** | 0.8895 | 0.9315 | 0.9100 | 0.8444 | 0.7625 | 0.8014 | `[[3429, 252], [426, 1368]]` | 1713.67 | **12.92** | **0.29** |
| **Polynomial** | `poly_d2_c2.0_r1_bal` | `sklearn.svm.SVC` | **0.8550** | 0.9024 | 0.9090 | 0.9057 | 0.8104 | 0.7982 | 0.8043 | `[[3346, 335], [362, 1432]]` | 1803.19 | 24.50 | 1.15 |

---

## 4. Analisis Peningkatan Performa ke > 0.85

1. **Mengapa Skor Meningkat dari 0.81 ke 0.857:**
   - Analisis sensitivitas dan *permutation importance* menunjukkan bahwa pembatalan kamar hotel didominasi oleh kombinasi non-linear antara `days_before_arrival`, `special_requests`, `booking_channel`, `nightly_rate_idr`, dan tren musiman `(arrival_year, arrival_month)`.
   - Penerapan *non-linear lead time bins*, pemetaan *special requests* tepat, rasio harga terhadap rata-rata segmen kamar/kanal, serta *out-of-fold target encoding* dan fitur interaksi pohon (`tree_prob`) berhasil memberikan representasi kontinu yang sangat mudah dipisahkan oleh *hyperplane* SVM.
2. **Efisiensi dan Keselarasan Numerik:**
   - `SVMFromScratch` mencapai keselarasan hampir mutlak dengan `sklearn.svm.SVC` (selisih Macro F1 hanya $\approx 0.0001 - 0.0007$).
   - Berkat representasi *Dual Coordinate Descent* pada ruang terproyeksi, waktu latih `SVMFromScratch` jauh lebih hemat:
     - Linear: **2,47 detik** vs 29,83 detik (12x lebih cepat).
     - RBF: **10,77 detik** vs 25,84 detik (2,4x lebih cepat).
     - Inferensi Linear pada validasi hanya **0,001 detik**.
