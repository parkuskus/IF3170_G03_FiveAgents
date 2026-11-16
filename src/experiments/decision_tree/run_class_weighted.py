import pandas as pd

if __package__:
    from .run_all import ROOT, run_weighted_cart, validation_indices
else:
    from run_all import ROOT, run_weighted_cart, validation_indices


def main():
    train_df = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    train_idx, valid_idx = validation_indices(train_df["is_canceled"])
    row, best, _ = run_weighted_cart(train_df, train_idx, valid_idx)
    print(f"Selected positive weight: {best['positive_weight']}")
    print(f"CV macro F1: {best['macro_f1']:.4f}")
    print(f"Holdout macro F1: {row['validation_macro_f1']:.4f}")
    print(f"Positive-class F1: {row['class_1_f1']:.4f}")
    print(f"Results: {ROOT / 'output' / 'experiments' / 'decision_tree' / 'CART_class_weight'}")


if __name__ == "__main__":
    main()
