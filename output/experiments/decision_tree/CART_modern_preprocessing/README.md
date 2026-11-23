# CART modern preprocessing

Same maximal CART, fold-local preprocessing, cost-complexity pruning, and threshold selection as the saved final method. Added missingness indicators and cyclic annual/weekly arrival-date features. No test data or submission is used.

Selected alpha `0.0001` and threshold **0.32**; OOF macro F1 **0.8306**; holdout macro F1 **0.8321** (positive-class F1 **0.7712**).
