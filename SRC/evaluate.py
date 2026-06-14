import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from SRC.train_neural import AirlineNet
import os

os.makedirs('Models/plots', exist_ok=True)

def evaluate_sklearn_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_metrics(name, y_pred, y_prob, y_test)

def evaluate_neural_model(name, model, X_test, y_test):
    model.eval()
    X_test_t = torch.FloatTensor(X_test.copy())
    with torch.no_grad():
        y_prob = model(X_test_t).numpy().flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(name, y_pred, y_prob, y_test)

def compute_metrics(name, y_pred, y_prob, y_test):
    metrics = {
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_prob)
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
    ax.plot(fpr, tpr, label=f'{name} (AUC={metrics["AUC-ROC"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'ROC Curve - {name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'Models/plots/roc_curve_{name.replace(" ", "_")}.png')
    plt.close()

    return metrics

def compare_models(results):
    import pandas as pd
    results_df = pd.DataFrame(results).set_index('Model')

    print("\n=== Model Comparison ===")
    print(results_df.round(4))

    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison - All Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('Models/plots/model_comparison.png')
    plt.close()

    best = results_df['F1'].idxmax()
    print(f"\n🏆 Best model (F1): {best}")

    with open('Models/evaluation_results.txt', 'w') as f:
        f.write("=== Model Comparison ===\n")
        f.write(results_df.round(4).to_string() + "\n\n")
        f.write(f"Best model (F1): {best}\n")

    return best, results_df

def save_best_model(best_name, rf_model, lr_model, nn_model, input_dim):
    if best_name == 'Random Forest':
        with open('Models/best_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        print("Best model saved as best_model.pkl ✓")
    elif best_name == 'Logistic Regression':
        with open('Models/best_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        print("Best model saved as best_model.pkl ✓")
    else:
        best_nn = AirlineNet(input_dim)
        best_nn.load_state_dict(torch.load('Models/neural_network.pt'))
        torch.save(best_nn.state_dict(), 'Models/best_model.pt')
        print("Best model saved as best_model.pt ✓")

def run():
    from SRC import preprocessing
    from SRC import train_classical, train_logistic, train_neural

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names) = preprocessing.run()

    rf_model  = train_classical.train(X_train, y_train, X_val, y_val, feature_names)
    lr_model  = train_logistic.train(X_train, y_train, X_val, y_val, feature_names)
    nn_model  = train_neural.train(X_train, y_train, X_val, y_val)

    results = []
    results.append(evaluate_sklearn_model('Random Forest', rf_model, X_test, y_test))
    results.append(evaluate_sklearn_model('Logistic Regression', lr_model, X_test, y_test))
    results.append(evaluate_neural_model('Neural Network', nn_model, X_test, y_test))

    best_name, results_df = compare_models(results)
    save_best_model(best_name, rf_model, lr_model, nn_model, X_test.shape[1])

    print("\nEvaluation complete ✓")

if __name__ == '__main__':
    run()