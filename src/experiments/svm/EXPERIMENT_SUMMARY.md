# Ringkasan Eksperimen SVM & Komparasi `scikit-learn` (Tahap 3)

## 1. Konfigurasi Split dan Prapemrosesan
- **Split Kontrak (`seed=42`, Stratified 80:20):**
  - `train_idx`: **21.898 sampel** (Kelas `0`: 14.723, Kelas `1`: 7.175)
  - `valid_idx`: **5.475 sampel** (Kelas `0`: 3.681, Kelas `1`: 1.794)
- **Prapemrosesan (`SVMPreprocessor`, fit hanya pada `train_idx`):**
  - Jumlah fitur akhir: **51 fitur** (imputasi median/modus, 8 flag indikator missing value, 10 atribut turunan termasuk `rate_per_guest_night` & `historical_cancel_ratio`, one-hot encoding, serta `StandardScaler`).

---

## 2. Hasil Eksperimen 19 Kandidat `SVMFromScratch` (Diurutkan berdasarkan Macro F1 Validasi)

| Rank | Variant | Kernel | $C$ | Degree | Gamma | Coef0 | Class Weight | Macro F1 | F1 (Kelas 0) | F1 (Kelas 1) | TN | FP | FN | TP | Support Vectors | Slack Valid | Fit Time (s) | Predict Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`poly_d3_c2.0_r1_bal`** | **poly** | **2.0** | **3** | **scale** | **1.0** | **balanced** | **0.8102** | **0.8722** | **0.7482** | **3168** | **513** | **415** | **1379** | **15277** | **2475.32** | **4.55** | **0.26** |
| 2 | `poly_d3_c1.0_r1_bal` | poly | 1.0 | 3 | scale | 1.0 | balanced | 0.8101 | 0.8732 | 0.7471 | 3184 | 497 | 428 | 1366 | 14685 | 2392.66 | 4.56 | 0.22 |
| **3** | **`rbf_c2.0_scale_bal`** | **rbf** | **2.0** | **3** | **scale** | **0.0** | **balanced** | **0.8048** | **0.8682** | **0.7415** | **3148** | **533** | **423** | **1371** | **15188** | **2401.44** | **3.78** | **0.16** |
| 4 | `rbf_c1.0_g0.03_bal` | rbf | 1.0 | 3 | 0.03 | 0.0 | balanced | 0.8015 | 0.8622 | 0.7409 | 3082 | 599 | 386 | 1408 | 14805 | 2430.77 | 4.11 | 0.18 |
| 5 | `poly_d3_c0.5_r1_bal` | poly | 0.5 | 3 | scale | 1.0 | balanced | 0.8012 | 0.8591 | 0.7433 | 3037 | 644 | 352 | 1442 | 13823 | 2419.96 | 7.04 | 0.22 |
| 6 | `poly_d3_c1.0_r1_none` | poly | 1.0 | 3 | scale | 1.0 | None | 0.7999 | 0.8816 | 0.7181 | 3399 | 282 | 631 | 1163 | 14126 | 2268.67 | 4.69 | 0.22 |
| 7 | `rbf_c2.0_g0.03_bal` | rbf | 2.0 | 3 | 0.03 | 0.0 | balanced | 0.7997 | 0.8607 | 0.7388 | 3073 | 608 | 387 | 1407 | 15190 | 2459.23 | 4.16 | 0.21 |
| 8 | `rbf_c1.0_scale_none` | rbf | 1.0 | 3 | scale | 0.0 | None | 0.7992 | 0.8771 | 0.7214 | 3332 | 349 | 585 | 1209 | 13826 | 2314.51 | 4.04 | 0.18 |
| 9 | `rbf_c2.0_g0.03_none` | rbf | 2.0 | 3 | 0.03 | 0.0 | None | 0.7991 | 0.8776 | 0.7206 | 3341 | 340 | 592 | 1202 | 14617 | 2293.16 | 4.26 | 0.17 |
| 10 | `poly_d2_c1.0_r1_bal` | poly | 1.0 | 2 | scale | 1.0 | balanced | 0.7971 | 0.8589 | 0.7353 | 3067 | 614 | 394 | 1400 | 14350 | 2469.13 | 4.09 | 0.16 |
| 11 | `rbf_c1.0_scale_bal` | rbf | 1.0 | 3 | scale | 0.0 | balanced | 0.7971 | 0.8589 | 0.7353 | 3067 | 614 | 394 | 1400 | 14708 | 2447.48 | 3.77 | 0.15 |
| 12 | `poly_d2_c1.0_r1_none` | poly | 1.0 | 2 | scale | 1.0 | None | 0.7947 | 0.8766 | 0.7127 | 3358 | 323 | 622 | 1172 | 13382 | 2330.22 | 5.47 | 0.13 |
| 13 | `rbf_c0.5_scale_bal` | rbf | 0.5 | 3 | scale | 0.0 | balanced | 0.7877 | 0.8491 | 0.7263 | 2997 | 684 | 381 | 1413 | 13565 | 2477.97 | 3.85 | 0.14 |
| 14 | `poly_d3_c1.0_r0_bal` | poly | 1.0 | 3 | scale | 0.0 | balanced | 0.7860 | 0.8481 | 0.7240 | 2995 | 686 | 387 | 1407 | 15802 | 2731.88 | 5.28 | 0.24 |
| **15** | **`linear_c0.1_none`** | **linear** | **0.1** | **1** | **scale** | **0.0** | **None** | **0.7653** | **0.8595** | **0.6711** | **3297** | **384** | **694** | **1100** | **15125** | **2616.96** | **1.15** | **0.00** |
| 16 | `linear_c0.1_bal` | linear | 0.1 | 1 | scale | 0.0 | balanced | 0.7609 | 0.8290 | 0.6927 | 2916 | 765 | 438 | 1356 | 15684 | 2852.05 | 1.00 | 0.00 |
| 17 | `linear_c1.0_bal` | linear | 1.0 | 1 | scale | 0.0 | balanced | 0.7564 | 0.8560 | 0.6568 | 3301 | 380 | 731 | 1063 | 18163 | 3149.38 | 0.88 | 0.00 |
| 18 | `linear_c1.0_none` | linear | 1.0 | 1 | scale | 0.0 | None | 0.7564 | 0.8560 | 0.6568 | 3301 | 380 | 731 | 1063 | 18163 | 3149.38 | 1.01 | 0.00 |
| 19 | `linear_c5.0_bal` | linear | 5.0 | 1 | scale | 0.0 | balanced | 0.7564 | 0.8560 | 0.6568 | 3301 | 380 | 731 | 1063 | 18163 | 3149.38 | 1.16 | 0.00 |

**Kandidat Terbaik Keseluruhan SVM Dari Nol:**
- **`poly_d3_c2.0_r1_bal`** (`kernel="poly"`, `C=2.0`, `degree=3`, `gamma="scale"`, `coef0=1.0`, `class_weight="balanced"`) dengan **Macro F1 = 0.8102**.

---

## 3. Komparasi Head-to-Head dengan `sklearn.svm.SVC`

| Kernel | Variant | Implementasi | Macro F1 | Prec 0 | Rec 0 | F1 0 | Prec 1 | Rec 1 | F1 1 | Confusion Matrix `[[TN, FP], [FN, TP]]` | Slack Valid | Fit Time (s) | Predict Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear** | `linear_c0.1_none` | `SVMFromScratch` | **0.7653** | 0.8261 | 0.8957 | 0.8595 | 0.7412 | 0.6132 | 0.6711 | `[[3297, 384], [694, 1100]]` | 2616.96 | **1.15** | **0.001** |
| **Linear** | `linear_c0.1_none` | `sklearn.svm.SVC` | 0.7601 | 0.8214 | 0.8973 | 0.8577 | 0.7400 | 0.5998 | 0.6626 | `[[3303, 378], [718, 1076]]` | 2553.08 | 21.61 | 1.453 |
| **Polynomial** | `poly_d3_c2.0_r1_bal` | `SVMFromScratch` | **0.8102** | 0.8842 | 0.8606 | 0.8722 | 0.7289 | 0.7687 | 0.7482 | `[[3168, 513], [415, 1379]]` | 2475.32 | **4.55** | **0.255** |
| **Polynomial** | `poly_d3_c2.0_r1_bal` | `sklearn.svm.SVC` | 0.8071 | 0.9038 | 0.8242 | 0.8622 | 0.6945 | 0.8200 | 0.7520 | `[[3034, 647], [323, 1471]]` | 2332.20 | 32.77 | 1.356 |
| **Gaussian (RBF)** | `rbf_c2.0_scale_bal` | `SVMFromScratch` | 0.8048 | 0.8815 | 0.8552 | 0.8682 | 0.7201 | 0.7642 | 0.7415 | `[[3148, 533], [423, 1371]]` | 2401.44 | **3.78** | **0.155** |
| **Gaussian (RBF)** | `rbf_c2.0_scale_bal` | `sklearn.svm.SVC` | **0.8084** | 0.9045 | 0.8256 | 0.8632 | 0.6965 | 0.8211 | 0.7536 | `[[3039, 642], [321, 1473]]` | 2291.38 | 28.60 | 7.369 |

---

## 4. Analisis Stabilitas Numerik, Performa, dan Biaya Komputasi

1. **Keselarasan Skor dengan `sklearn.svm.SVC`:**
   - Selisih Macro F1 antara `SVMFromScratch` dan `sklearn.svm.SVC` sangat kecil pada ketiga kernel: **+0.0052** pada Linear (`0.7653` vs `0.7601`), **+0.0031** pada Polynomial (`0.8102` vs `0.8071`), dan **-0.0036** pada Gaussian/RBF (`0.8048` vs `0.8084`).
   - Kernel non-linear (**Polynomial derajat 3** dan **Gaussian/RBF**) meningkatkan Macro F1 secara signifikan ($\sim +4,5\%$) dibanding kernel Linear karena interaksi non-linear antarfitur (seperti `days_before_arrival`, `special_requests`, `historical_cancel_ratio`, dan `rate_per_guest_night`).
   - Penggunaan `class_weight="balanced"` ($C_i = C \cdot \frac{N}{2 N_{y_i}}$) terbukti krusial untuk mengangkat *recall* kelas minoritas (`is_canceled = 1`) dari ~64,8% menjadi ~76,9%, sehingga mendongkrak Macro F1 dari `0.7999` menjadi `0.8102`.
2. **Stabilitas Numerik:**
   - Standardisasi (`StandardScaler`) menjaga varians setiap fitur $= 1$, sehingga $\gamma_{\text{scale}} = \frac{1}{D \cdot \text{Var}(X)} \approx \frac{1}{51} \approx 0.0196$. Tanpa standardisasi, fitur `nightly_rate_idr` ($\sim 10^6$) menyebabkan $\mathbf{x}_i^\top \mathbf{x}_j \sim 10^{12}$ yang membuat kernel polinomial $d=3$ mengalami *floating-point overflow* ($\sim 10^{36}$) dan kernel RBF mengalami *underflow* ($\exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) \to 0$ untuk semua $i \neq j$).
   - Pada kernel Polynomial, suku inhomogen `coef0 = 1.0` memberikan Macro F1 jauh lebih tinggi (`0.8101`) dibanding `coef0 = 0.0` (`0.7860`) karena mempertahankan kontribusi suku derajat rendah (linear dan kuadratik) bersama suku kubik.
3. **Biaya Komputasi & Memori:**
   - Dengan metode *Dual Coordinate Descent* di ruang RKHS terproyeksi (`n_components=1000`), kebutuhan memori berhasil ditekan menjadi **~8,91 MB** (dibandingkan matriks Gram penuh $21.898 \times 21.898$ yang membutuhkan **3,84 GB** RAM).
   - Waktu pelatihan `SVMFromScratch` mencapai **1,15 detik** (Linear), **4,55 detik** (Polynomial), dan **3,78 detik** (RBF), atau sekitar **6–18 kali lebih cepat** dibandingkan solver SMO penuh `sklearn.svm.SVC` (`21,61s` – `32,77s`), serta waktu prediksi pada kernel Linear mendekati **0,001 detik** karena menggunakan evaluasi primal langsung $\mathbf{w}^\top \mathbf{x} + b$.
