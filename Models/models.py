import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from Models import preprocessing
import os

os.makedirs('Models/plots', exist_ok=True)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Satisfied', 'Satisfied'],
                yticklabels=['Not Satisfied', 'Satisfied'])
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'Models/plots/confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f'{name} (AUC = {metrics["AUC-ROC"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'ROC Curve - {name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'Models/plots/roc_curve_{name.replace(" ", "_")}.png')
    plt.close()

    return metrics

def run():
    # Χρησιμοποιούμε το καθαρό dataset από το preprocessing
    df = preprocessing.run()

    X = df.drop(columns=['liked'])
    y = df['liked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results.append(evaluate_model('Logistic Regression', lr, X_test, y_test))

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate_model('Random Forest', rf, X_test, y_test))

    results_df = pd.DataFrame(results).set_index('Model')
    print("\n=== Model Comparison ===")
    print(results_df.round(4))

    with open('Models/model_results.txt', 'w') as f:
        f.write("=== Model Comparison ===\n")
        f.write(results_df.round(4).to_string() + "\n\n")
        best = results_df['F1'].idxmax()
        f.write(f"Best model (F1): {best}\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison - All Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Models/plots/model_comparison.png')
    plt.close()

    feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
    top20 = feat_imp.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top20.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Random Forest - Top 20 Feature Importance')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('Models/plots/feature_importance.png')
    plt.close()

    print("\nModels saved ✓")