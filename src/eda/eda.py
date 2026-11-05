"""EDA deskriptif untuk data mentah; tidak melatih model atau mengubah CSV asal."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "eda"
CATEGORIES = ["meal_package", "room_category", "booking_channel"]
COUNTS = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "days_before_arrival", "prior_cancellations", "prior_completed_bookings",
    "special_requests", "nightly_rate_idr",
]


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ])


def pct(count, size):
    return f"{count} ({100 * count / size:.2f}%)" if size else "0 (0.00%)"


def invalid_dates(frame):
    dates = frame[["arrival_year", "arrival_month", "arrival_day"]].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    return pd.to_datetime(dates, errors="coerce").isna().sum()


def histogram_limit(values):
    upper = values.quantile(0.99)
    return values.max() if upper <= values.min() else upper


def default_output_dir():
    return OUTPUT


def draw_charts(train, test, out):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    target = train["is_canceled"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Tidak batal (0)", "Batal (1)"], target, color=["#457b9d", "#e76f51"])
    ax.set(ylabel="Jumlah pemesanan", title="Distribusi target pada train")
    for i, value in enumerate(target):
        ax.text(i, value, f"{value:,} ({value / len(train):.1%})", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "distribusi_target.png")
    plt.close(fig)

    missing = pd.DataFrame({
        "Train": train.isna().mean().mul(100),
        "Test": test.isna().mean().mul(100),
    }).dropna().sort_values("Train", ascending=True)
    missing = missing[missing.max(axis=1) > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, len(missing) * 0.42)))
        missing.plot.barh(ax=ax, color=["#457b9d", "#e76f51"])
        ax.set(xlabel="Baris dengan nilai kosong (%)", title="Kekosongan per kolom")
        fig.tight_layout()
        fig.savefig(out / "nilai_kosong.png")
        plt.close(fig)

    numbers = [c for c in COUNTS if c in train]
    if numbers:
        fig, axes = plt.subplots(int(np.ceil(len(numbers) / 3)), 3, figsize=(13, 3 * int(np.ceil(len(numbers) / 3))))
        for ax, col in zip(axes.flat, numbers):
            values = train[col].dropna()
            upper = histogram_limit(values)
            ax.hist(values.clip(upper=upper), bins=25, color="#457b9d")
            ax.set(title=col, ylabel="Jumlah")
            if values.max() > upper:
                ax.set_xlabel("Nilai di atas p99 dijadikan p99 pada grafik")
        for ax in list(axes.flat)[len(numbers):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "distribusi_numerik.png")
        plt.close(fig)

    cats = [c for c in CATEGORIES if c in train]
    if cats:
        fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4))
        for ax, col in zip(np.atleast_1d(axes), cats):
            counts = train[col].fillna("(kosong)").value_counts()
            counts.sort_values().plot.barh(ax=ax, color="#457b9d")
            ax.set(title=col, xlabel="Jumlah train")
        fig.tight_layout()
        fig.savefig(out / "distribusi_kategori.png")
        plt.close(fig)

    if cats:
        fig, axes = plt.subplots(1, len(cats), figsize=(5 * len(cats), 4))
        for ax, col in zip(np.atleast_1d(axes), cats):
            rates = train.groupby(train[col].fillna("(kosong)"))["is_canceled"].agg(["mean", "size"])
            rates.sort_values("mean")["mean"].mul(100).plot.barh(ax=ax, color="#e76f51")
            ax.set(title=col, xlabel="Tingkat pembatalan train (%)", xlim=(0, 100))
        fig.tight_layout()
        fig.savefig(out / "pembatalan_per_kategori.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    months = range(1, 13)
    axes[0].plot(months, train.arrival_month.value_counts(normalize=True).reindex(months, fill_value=0).mul(100), marker="o", label="Train")
    axes[0].plot(months, test.arrival_month.value_counts(normalize=True).reindex(months, fill_value=0).mul(100), marker="o", label="Test")
    axes[0].set(xlabel="Bulan kedatangan", ylabel="Proporsi baris (%)", xticks=list(months))
    axes[0].legend()
    axes[1].plot(months, train.groupby("arrival_month").is_canceled.mean().reindex(months).mul(100), marker="o", color="#e76f51")
    axes[1].set(xlabel="Bulan kedatangan", ylabel="Pembatalan train (%)", xticks=list(months), ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(out / "kedatangan_bulanan.png")
    plt.close(fig)


def generate(train, test, out, sample=None):
    """Simpan laporan dan grafik hanya ke out, tanpa mengubah train/test."""
    out.mkdir(parents=True, exist_ok=True)
    if "is_canceled" not in train or "is_canceled" in test:
        raise ValueError("Target harus hanya ada pada train")
    if set(train.columns) - {"is_canceled"} != set(test.columns):
        raise ValueError("Kolom fitur train dan test berbeda")

    features = [c for c in train if c not in ("id", "is_canceled")]
    overlap = train.loc[train["id"].isin(test["id"]), "id"].nunique()
    duplicate_features = train.duplicated(subset=features, keep=False)
    conflicts = train.loc[duplicate_features].groupby(features, dropna=False)["is_canceled"].nunique()
    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    lines = [
        "# Eksplorasi data pemesanan penginapan",
        "",
        "Dihasilkan oleh `python src/eda/eda.py` dari CSV mentah. Tidak ada imputasi, penggabungan train dan test, atau pelatihan model. Statistik target hanya memakai train. Semua persentase per kolom memakai jumlah baris berkas terkait sebagai penyebut.",
        "",
        f"Terdapat **{len(train):,} baris latih dan {len(test):,} baris test**. Train memiliki {train.shape[1]} kolom, sedangkan test memiliki {test.shape[1]} kolom tanpa `is_canceled`.",
        "",
        "## Integritas data",
        "",
        table(["Pemeriksaan", "Train", "Test"], [
            ("ID kosong", train["id"].isna().sum(), test["id"].isna().sum()),
            ("ID duplikat pada train", train["id"].duplicated().sum(), "-"),
            ("ID duplikat pada test", "-", test["id"].duplicated().sum()),
            ("ID yang muncul di kedua berkas", overlap, overlap),
            ("Baris identik termasuk ID", train.duplicated().sum(), test.duplicated().sum()),
            ("Baris fitur berulang tanpa ID/target", train.duplicated(subset=features).sum(), test.duplicated(subset=features).sum()),
            ("Baris test dengan fitur sama seperti train (hash)", "-", test_hash.isin(train_hash).sum()),
            ("Kelompok fitur berulang dengan target bertentangan", (conflicts > 1).sum(), "tidak berlabel"),
        ]),
        "",
        "Jumlah baris fitur berulang adalah hitungan setelah kemunculan pertama. Kesamaan train–test dihitung dari hash semua fitur termasuk nilai kosong, bukan dari ID. Periksa baris asli sebelum menganggapnya duplikat yang harus dibuang.",
        "",
    ]
    if sample is not None:
        lines += ["## Templat submisi", "", table(["Pemeriksaan", "Hasil"], [
            ("Header sesuai `id,is_canceled`", "ya" if list(sample.columns) == ["id", "is_canceled"] else "tidak"),
            ("Jumlah baris sama dengan test", "ya" if len(sample) == len(test) else "tidak"),
            ("Urutan ID sama dengan test", "ya" if sample["id"].reset_index(drop=True).equals(test["id"].reset_index(drop=True)) else "tidak"),
            ("Label contoh hanya 0/1", "ya" if sample["is_canceled"].isin([0, 1]).all() else "tidak"),
        ]), "", "Label pada templat hanya contoh, bukan jawaban test.", ""]
    lines += ["## Target dan nilai kosong", ""]
    target = train["is_canceled"].value_counts(dropna=False).sort_index()
    lines += [table(["is_canceled", "Jumlah", "Persentase train"], [(str(k), v, f"{v / len(train):.2%}") for k, v in target.items()]), "", "![Target](distribusi_target.png)", ""]
    lines += [table(["Kolom", "Train kosong", "Test kosong"], [
        (f"`{col}`", pct(train[col].isna().sum(), len(train)), pct(test[col].isna().sum(), len(test)))
        for col in train if train[col].isna().any() or test.get(col, pd.Series(dtype=float)).isna().any()
    ]), "", f"Baris dengan setidaknya satu fitur kosong adalah {pct(train[features].isna().any(axis=1).sum(), len(train))} pada train dan {pct(test[features].isna().any(axis=1).sum(), len(test))} pada test.", "", "![Nilai kosong](nilai_kosong.png)", ""]
    missing_rates = [(c, int(train[c].isna().sum()), train.loc[train[c].isna(), "is_canceled"].mean(), train.loc[train[c].notna(), "is_canceled"].mean()) for c in features if train[c].isna().any()]
    lines += ["Tingkat pembatalan menurut kelengkapan fitur (deskriptif, bukan bukti sebab-akibat).", "", table(["Fitur", "Kosong", "Batal jika kosong", "Batal jika terisi"], [
        (c, n, f"{missing:.2%}", f"{present:.2%}") for c, n, missing, present in missing_rates
    ]), "", "## Statistik fitur numerik", ""]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    describe = train[numeric].describe(percentiles=[.25, .5, .75, .95, .99]).T
    lines += [table(["Fitur", "Terisi", "Rata-rata", "Min", "p25", "Median", "p75", "p95", "p99", "Maks"], [
        (c, int(row["count"]), *(f"{row[k]:,.2f}" for k in ("mean", "min", "25%", "50%", "75%", "95%", "99%", "max")))
        for c, row in describe.iterrows()
    ]), "", "Grafik memotong tampilan di persentil ke-99 supaya ekor panjang tidak menyembunyikan mayoritas observasi. Jika persentil ke-99 sama dengan minimum, batas grafik memakai maksimum agar nilai langka tetap terlihat. Tabel di atas selalu menampilkan maksimum asli.", "", "![Numerik](distribusi_numerik.png)", ""]
    lines += ["## Kategori dan tingkat pembatalan", ""]
    for col in CATEGORIES:
        if col not in train:
            continue
        grouped = train.groupby(train[col].fillna("(kosong)"))["is_canceled"].agg(["size", "mean"])
        test_counts = test[col].fillna("(kosong)").value_counts()
        lines += [f"### {col}", "", table(["Nilai", "Train", "Test", "Batal pada train"], [
            (str(value), pct(int(row["size"]), len(train)), pct(test_counts.get(value, 0), len(test)), f"{row['mean']:.2%}")
            for value, row in grouped.sort_values("size", ascending=False).iterrows()
        ]), ""]
    lines += ["![Kategori](distribusi_kategori.png)", "", "![Pembatalan menurut kategori](pembatalan_per_kategori.png)", "", "## Kedatangan menurut tahun dan bulan", ""]
    for col in ("arrival_year", "arrival_month"):
        rates = train.groupby(col).is_canceled.agg(["size", "mean"])
        test_counts = test[col].value_counts()
        lines += [f"### {col}", "", table(["Nilai", "Train", "Test", "Batal pada train"], [
            (value, int(row["size"]), int(test_counts.get(value, 0)), f"{row['mean']:.2%}")
            for value, row in rates.iterrows()
        ]), ""]
    lines += ["![Kedatangan bulanan](kedatangan_bulanan.png)", "", "## Pemeriksaan nilai yang perlu ditinjau", ""]
    checks = {
        "Tanggal kedatangan tidak sah": (invalid_dates(train), invalid_dates(test)),
        "Tidak ada malam menginap": ((train.weekend_nights + train.weekday_nights == 0).sum(), (test.weekend_nights + test.weekday_nights == 0).sum()),
        "Dewasa dan anak sama-sama nol (baris lengkap)": (((train.num_adults == 0) & (train.num_children == 0)).sum(), ((test.num_adults == 0) & (test.num_children == 0)).sum()),
        "Tarif per malam nol (mungkin complimentary)": ((train.nightly_rate_idr == 0).sum(), (test.nightly_rate_idr == 0).sum()),
        "Nilai hitungan atau tarif negatif": (train[[c for c in COUNTS if c in train]].lt(0).any(axis=1).sum(), test[[c for c in COUNTS if c in test]].lt(0).any(axis=1).sum()),
    }
    lines += [table(["Kondisi", "Train", "Test"], [(label, a, b) for label, (a, b) in checks.items()]), "", "Nilai nol dan nilai ekstrem belum tentu salah. `nightly_rate_idr = 0` dapat menandai pemesanan gratis menurut deskripsi data. Tanggal yang tidak sah perlu diperiksa sebelum membuat fitur tanggal.", "", "## Train dibanding test", ""]
    lines += [table(["Numerik", "Median train", "Median test", "p95 train", "p95 test", "Selisih kosong (poin %)"], [
        (c, f"{train[c].median():,.2f}", f"{test[c].median():,.2f}", f"{train[c].quantile(.95):,.2f}", f"{test[c].quantile(.95):,.2f}", f"{100 * (test[c].isna().mean() - train[c].isna().mean()):+.2f}")
        for c in numeric
    ]), ""]
    drift = []
    for c in CATEGORIES:
        if c not in train:
            continue
        a = train[c].fillna("(kosong)").value_counts(normalize=True)
        b = test[c].fillna("(kosong)").value_counts(normalize=True)
        levels = a.index.union(b.index)
        drift.append((c, f"{(a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).abs().sum() / 2:.3f}", ", ".join(map(str, b.index.difference(a.index))) or "-"))
    lines += [table(["Kategori", "Jarak proporsi (0–1)", "Nilai hanya di test"], drift), "", "Jarak proporsi di atas adalah setengah jumlah selisih absolut proporsi kategori, bukan uji signifikansi. Perbedaan ringkasan tidak otomatis berarti data mengalami drift yang merusak model.", "", "## Hubungan deskriptif dengan pembatalan", ""]
    numeric_corr = train[numeric + ["is_canceled"]].corr(method="spearman")["is_canceled"].drop("is_canceled").sort_values(key=lambda x: x.abs(), ascending=False)
    lines += [table(["Fitur", "Spearman dengan target"], [(c, f"{value:+.3f}" if pd.notna(value) else "tidak terdefinisi") for c, value in numeric_corr.items()]), "", "Korelasi ini tidak menunjukkan sebab-akibat. Angka dari fitur kategori tidak diperlakukan sebagai urutan.", ""]
    for col in ["days_before_arrival", "nightly_rate_idr", "special_requests", "weekend_nights"]:
        if col not in train:
            continue
        bins = pd.qcut(train[col], q=5, duplicates="drop")
        rates = train.groupby(bins, observed=True)["is_canceled"].agg(["size", "mean"])
        lines += [f"### Tingkat pembatalan menurut {col}", "", table(["Rentang kuantil train", "Baris", "Batal"], [
            (str(interval), int(row["size"]), f"{row['mean']:.2%}") for interval, row in rates.iterrows()
        ]), ""]
    lines += ["## Implikasi untuk eksperimen", "", "1. Pertahankan `id` sebagai string untuk submisi, tetapi keluarkan dari fitur. Periksa duplikat fitur dan tanggal tidak sah sebelum menetapkan aturan cleaning.", "2. Tangani nilai kosong dan kategori langka berdasarkan bagian latih saja. Siapkan perilaku untuk kategori yang tidak terlihat saat pelatihan.", "3. Gunakan split validasi yang sama sesuai README akar. EDA ini memakai seluruh train hanya untuk memahami data mentah; angka deskriptif di sini jangan digunakan untuk memilih transformasi dengan melihat validasi atau test.", ""]
    draw_charts(train, test, out)
    (out / "laporan.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate(
        pd.read_csv(ROOT / "data/train.csv", dtype={"id": str}),
        pd.read_csv(ROOT / "data/test.csv", dtype={"id": str}),
        default_output_dir(),
        pd.read_csv(ROOT / "data/sample_submission.csv", dtype={"id": str}),
    )
