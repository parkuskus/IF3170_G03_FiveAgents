from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.svm import SVC

from preprocessing import SVMPreprocessor, load_and_prepare_split
from svm_scratch import SVMFromScratch


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))
    prec, rec, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "macro_f1": macro_f1,
        "prec_0": float(prec[0]),
        "rec_0": float(rec[0]),
        "f1_0": float(f1_cls[0]),
        "prec_1": float(prec[1]),
        "rec_1": float(rec[1]),
        "f1_1": float(f1_cls[1]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def calculate_svc_slack(svc_model: SVC, x: np.ndarray, y_raw: np.ndarray) -> float:
    y_pm = np.where(y_raw == 1, 1.0, -1.0)
    scores = svc_model.decision_function(x)
    margins = y_pm * scores
    return float(np.sum(np.maximum(0.0, 1.0 - margins)))


def get_candidate_configs() -> list[dict]:
    return [
        # kernel linear
        {"variant": "linear_c0.5_none", "kernel": "linear", "C": 0.5, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": None},
        {"variant": "linear_c1.0_none", "kernel": "linear", "C": 1.0, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": None},
        {"variant": "linear_c1.0_bal", "kernel": "linear", "C": 1.0, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "linear_c2.0_none", "kernel": "linear", "C": 2.0, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": None},
        {"variant": "linear_c2.0_bal", "kernel": "linear", "C": 2.0, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "linear_c4.0_bal", "kernel": "linear", "C": 4.0, "degree": 1, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        # kernel polynomial
        {"variant": "poly_d2_c2.0_r1_bal", "kernel": "poly", "C": 2.0, "degree": 2, "gamma": "scale", "coef0": 1.0, "class_weight": "balanced"},
        {"variant": "poly_d3_c1.0_r1_none", "kernel": "poly", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0, "class_weight": None},
        {"variant": "poly_d3_c1.0_r1_bal", "kernel": "poly", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 1.0, "class_weight": "balanced"},
        {"variant": "poly_d3_c2.0_r1_bal", "kernel": "poly", "C": 2.0, "degree": 3, "gamma": "scale", "coef0": 1.0, "class_weight": "balanced"},
        {"variant": "poly_d3_c3.0_r1_bal", "kernel": "poly", "C": 3.0, "degree": 3, "gamma": "scale", "coef0": 1.0, "class_weight": "balanced"},
        {"variant": "poly_d3_c4.0_r1_bal", "kernel": "poly", "C": 4.0, "degree": 3, "gamma": "scale", "coef0": 1.0, "class_weight": "balanced"},
        # kernel gaussian / rbf
        {"variant": "rbf_c1.0_scale_none", "kernel": "rbf", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 0.0, "class_weight": None},
        {"variant": "rbf_c1.0_scale_bal", "kernel": "rbf", "C": 1.0, "degree": 3, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "rbf_c2.0_scale_bal", "kernel": "rbf", "C": 2.0, "degree": 3, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "rbf_c3.0_scale_bal", "kernel": "rbf", "C": 3.0, "degree": 3, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "rbf_c4.0_scale_bal", "kernel": "rbf", "C": 4.0, "degree": 3, "gamma": "scale", "coef0": 0.0, "class_weight": "balanced"},
        {"variant": "rbf_c4.0_g0.01_bal", "kernel": "rbf", "C": 4.0, "degree": 3, "gamma": 0.01, "coef0": 0.0, "class_weight": "balanced"},
    ]


def run_all_experiments():
    exp_dir = Path(__file__).resolve().parent
    repo_root = exp_dir.parents[2]
    bundle = load_and_prepare_split(repo_root / "data", seed=42)

    x_train = bundle["X_train"]
    y_train = bundle["y_train"]
    x_valid = bundle["X_valid"]
    y_valid = bundle["y_valid"]

    candidates = get_candidate_configs()
    rows = []

    print(f"menjalankan eksperimen {len(candidates)} kandidat svm dari nol dengan pipeline fitur baru...")
    for cfg in candidates:
        model = SVMFromScratch(
            C=cfg["C"],
            kernel=cfg["kernel"],
            degree=cfg["degree"],
            gamma=cfg["gamma"],
            coef0=cfg["coef0"],
            class_weight=cfg["class_weight"],
            max_iter=45,
            n_components=1400,
            random_state=42,
        )
        model.fit(x_train, y_train)

        # evaluasi default threshold 0
        y_pred_def = model.predict(x_valid, threshold=0.0)
        metrics_def = evaluate_predictions(y_valid, y_pred_def)

        # evaluasi threshold ter-tune
        th_tuned = model.tune_threshold(x_valid, y_valid)
        y_pred_tuned = model.predict(x_valid, threshold=th_tuned)
        metrics_tuned = evaluate_predictions(y_valid, y_pred_tuned)

        slack_valid = model.calculate_slack(x_valid, y_valid)
        sv_bytes = 0 if model.support_vectors_ is None else model.support_vectors_.nbytes
        mem_mb = (x_train.nbytes + sv_bytes) / (1024 ** 2)

        row = {
            **cfg,
            "class_weight": str(cfg["class_weight"]),
            "seed": 42,
            "default_macro_f1": round(metrics_def["macro_f1"], 4),
            "macro_f1": round(metrics_tuned["macro_f1"], 4),
            "best_threshold": round(th_tuned, 3),
            "prec_0": round(metrics_tuned["prec_0"], 4),
            "rec_0": round(metrics_tuned["rec_0"], 4),
            "f1_0": round(metrics_tuned["f1_0"], 4),
            "prec_1": round(metrics_tuned["prec_1"], 4),
            "rec_1": round(metrics_tuned["rec_1"], 4),
            "f1_1": round(metrics_tuned["f1_1"], 4),
            "tn": metrics_tuned["tn"],
            "fp": metrics_tuned["fp"],
            "fn": metrics_tuned["fn"],
            "tp": metrics_tuned["tp"],
            "n_support": model.n_support_,
            "slack_train": round(model.slack_total, 2),
            "slack_valid": round(slack_valid, 2),
            "fit_time_sec": round(model.fit_time_sec_, 3),
            "predict_time_sec": round(model.predict_time_sec_, 3),
            "mem_mb": round(mem_mb, 2),
        }
        rows.append(row)
        print(
            f"  {cfg['variant']:<22} | default_f1={metrics_def['macro_f1']:.4f} "
            f"| tuned_f1={metrics_tuned['macro_f1']:.4f} (th={th_tuned:+.2f}) | "
            f"fit={model.fit_time_sec_:.2f}s | sv={model.n_support_}"
        )

    df_exp = pd.DataFrame(rows)
    df_exp = df_exp.sort_values(by=["macro_f1", "variant"], ascending=[False, True]).reset_index(drop=True)
    df_exp.to_csv(exp_dir / "svm_experiment_results.csv", index=False)

    # komparasi konfigurasi terbaik tiap kernel dengan sklearn.svm.SVC
    comp_rows = []
    print("\nkomparasi konfigurasi terbaik tiap kernel dengan sklearn.svm.SVC...")
    for kern in ["linear", "poly", "rbf"]:
        sub = df_exp[df_exp["kernel"] == kern].sort_values(by=["macro_f1", "variant"], ascending=[False, True])
        best = sub.iloc[0].to_dict()

        comp_rows.append({
            "implementation": "from_scratch",
            "kernel": kern,
            "variant": best["variant"],
            "C": best["C"],
            "degree": best["degree"],
            "gamma": best["gamma"],
            "coef0": best["coef0"],
            "class_weight": best["class_weight"],
            "macro_f1": best["macro_f1"],
            "prec_0": best["prec_0"],
            "rec_0": best["rec_0"],
            "f1_0": best["f1_0"],
            "prec_1": best["prec_1"],
            "rec_1": best["rec_1"],
            "f1_1": best["f1_1"],
            "tn": best["tn"],
            "fp": best["fp"],
            "fn": best["fn"],
            "tp": best["tp"],
            "n_support": best["n_support"],
            "slack_train": best["slack_train"],
            "slack_valid": best["slack_valid"],
            "fit_time_sec": best["fit_time_sec"],
            "predict_time_sec": best["predict_time_sec"],
        })

        cw = None if best["class_weight"] == "None" else best["class_weight"]
        gamma_arg = best["gamma"]
        if isinstance(gamma_arg, str) and gamma_arg not in ("scale", "auto"):
            gamma_arg = float(gamma_arg)

        svc = SVC(
            C=float(best["C"]),
            kernel=kern,
            degree=int(best["degree"]),
            gamma=gamma_arg,
            coef0=float(best["coef0"]),
            class_weight=cw,
            random_state=42,
        )
        t0 = time.perf_counter()
        svc.fit(x_train, y_train)
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        svc_scores = svc.decision_function(x_valid)
        pred_t = time.perf_counter() - t1

        # cari threshold terbaik untuk sklearn svc juga agar adil
        best_svc_f1, best_svc_th = -1.0, 0.0
        for th in np.linspace(-1.0, 1.0, 101):
            f1 = f1_score(y_valid, (svc_scores >= th).astype(int), average="macro", zero_division=0)
            if f1 > best_svc_f1:
                best_svc_f1 = f1
                best_svc_th = th

        svc_pred = (svc_scores >= best_svc_th).astype(int)
        svc_metrics = evaluate_predictions(y_valid, svc_pred)
        svc_slack_tr = calculate_svc_slack(svc, x_train, y_train)
        svc_slack_va = calculate_svc_slack(svc, x_valid, y_valid)

        comp_rows.append({
            "implementation": "scikit_learn_SVC",
            "kernel": kern,
            "variant": best["variant"],
            "C": best["C"],
            "degree": best["degree"],
            "gamma": best["gamma"],
            "coef0": best["coef0"],
            "class_weight": best["class_weight"],
            **svc_metrics,
            "n_support": int(len(svc.support_)),
            "slack_train": round(svc_slack_tr, 2),
            "slack_valid": round(svc_slack_va, 2),
            "fit_time_sec": round(fit_t, 3),
            "predict_time_sec": round(pred_t, 3),
        })
        print(
            f"  [{kern}] scratch macro_f1={best['macro_f1']:.4f} ({best['fit_time_sec']:.2f}s) "
            f"vs sklearn macro_f1={svc_metrics['macro_f1']:.4f} ({fit_t:.2f}s)"
        )

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(exp_dir / "svm_sklearn_comparison.csv", index=False)

    winner = df_exp.iloc[0]
    print(f"\npemenang keseluruhan svm dari nol: {winner['variant']} (macro_f1={winner['macro_f1']:.4f})")
    return df_exp, df_comp


if __name__ == "__main__":
    run_all_experiments()
