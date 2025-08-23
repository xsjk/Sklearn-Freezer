# Sklearn-Freezer

**Sklearn-Freezer** is a Python project designed to optimize scikit-learn classifier inference by compiling their `predict` or `predict_proba` function into a static C extension. This project targets reducing the overhead of single-sample predictions, addressing scikit-learn's performance bottleneck for real-time or single-sample inference, which traditionally is better suited for batch inference.
