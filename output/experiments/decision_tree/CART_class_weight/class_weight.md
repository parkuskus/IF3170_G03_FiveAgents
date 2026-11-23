# CART class weighting

Positive-class weight is selected by 5-fold stratified CV on the training partition; each fold fits preprocessing and a weighted CART from scratch. The shared validation split is used only once for final evaluation. No submission is written.

Selected positive class weight: **1.0**; CV macro F1: **0.8068**; mean leaves: **117.6**.

Holdout macro F1: **0.8127**; positive-class F1: **0.7447**; leaves: **114**. Candidate scores are in `class_weight.csv`.
