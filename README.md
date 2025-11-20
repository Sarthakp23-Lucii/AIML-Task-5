# Heart Disease Prediction (Decision Tree & Random Forest)

This project applies machine learning models to the **Heart Disease Dataset** to predict whether a person is likely to have heart disease. The workflow includes model training, evaluation, visualization, and feature importance analysis.

## 🔧 Steps Performed
1. **Train a Decision Tree Classifier**
   - Fit on training data
   - Visualize the full decision tree (`tree_full.png`)

2. **Analyze Overfitting**
   - Compare performance of trees with different `max_depth`
   - Choose optimal depth based on accuracy

3. **Train a Random Forest**
   - Compare accuracy with Decision Tree
   - More stable and less prone to overfitting

4. **Feature Importances**
   - Extract top features from Random Forest
   - Visualized in `feature_importances_rf.png`

5. **Cross-Validation**
   - 5-fold Stratified CV for reliable performance estimation

## 📁 Files Generated
- `tree_full.png` – Full decision tree
- `tree_limited.png` – Depth-limited tree
- `feature_importances_rf.png` – Feature importance plot

## 🛠 Requirements
Install necessary libraries:
```bash
pip install pandas scikit-learn matplotlib
