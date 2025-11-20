import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(path='heart.csv'):
    df = pd.read_csv(path)
    print("Loaded data shape:", df.shape)
    print("Columns:", list(df.columns))
    return df

def prepare_data(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    return dt

def plot_and_save_tree(model, feature_names, class_names=('No','Yes'), filename='tree.png', figsize=(20,10)):
    plt.figure(figsize=figsize)
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved tree plot to {filename}")
    plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test, name='Model'):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"--- {name} ---")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, y_pred_test))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_test))
    print()
    return train_acc, test_acc

def plot_feature_importances(importances, feature_names, filename='feature_importances.png'):
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved feature importances to {filename}")
    plt.show()

def cross_validate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Cross-validation (cv={cv}) accuracies: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def main():
    df = load_data('heart.csv')
    X, y = prepare_data(df, target_col='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    dt_full = train_decision_tree(X_train, y_train, max_depth=None, random_state=42)
    evaluate_model(dt_full, X_train, y_train, X_test, y_test, name='Decision Tree (full)')
    plot_and_save_tree(dt_full, feature_names=X.columns, class_names=['No Disease','Disease'], filename='tree_full.png', figsize=(20,10))

    print("Depth tuning (test accuracies):")
    depths = [2, 3, 4, 5, 6, 8, 10]
    depth_results = {}
    for d in depths:
        dt_d = train_decision_tree(X_train, y_train, max_depth=d, random_state=42)
        _, test_acc = evaluate_model(dt_d, X_train, y_train, X_test, y_test, name=f'Decision Tree (max_depth={d})')
        depth_results[d] = test_acc

    print("Summary of depth -> test accuracy:")
    for d, acc in depth_results.items():
        print(f" depth={d:2d} -> test_acc={acc:.4f}")
    print()

    best_depth = max(depth_results, key=lambda k: depth_results[k])
    print(f"Best depth from tried set: {best_depth} (test acc {depth_results[best_depth]:.4f})")
    dt_best = train_decision_tree(X_train, y_train, max_depth=best_depth, random_state=42)
    evaluate_model(dt_best, X_train, y_train, X_test, y_test, name=f'Decision Tree (chosen depth={best_depth})')
    plot_and_save_tree(dt_best, feature_names=X.columns, class_names=['No Disease','Disease'], filename='tree_limited.png', figsize=(14,8))

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_train, y_train, X_test, y_test, name='Random Forest (200 trees)')

    dt_full_test_acc = accuracy_score(y_test, dt_full.predict(X_test))
    dt_best_test_acc = accuracy_score(y_test, dt_best.predict(X_test))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

    print("Comparison of test accuracies:")
    print(f" Decision Tree (full)      : {dt_full_test_acc:.4f}")
    print(f" Decision Tree (depth={best_depth}): {dt_best_test_acc:.4f}")
    print(f" Random Forest (200)      : {rf_test_acc:.4f}")
    print()

    importances = rf.feature_importances_
    print("Feature importances (Random Forest):")
    for name, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:12s}: {imp:.4f}")

    plot_feature_importances(importances, X.columns, filename='feature_importances_rf.png')

    print("Cross-validating models (5-fold Stratified):")
    print("- Decision Tree (depth chosen)")
    cross_validate_model(DecisionTreeClassifier(max_depth=best_depth, random_state=42), X, y, cv=5)
    print("- Random Forest (200 estimators)")
    cross_validate_model(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5)

    print("\nAll done. Check saved images: tree_full.png, tree_limited.png, feature_importances_rf.png")

if __name__ == "__main__":
    main()
