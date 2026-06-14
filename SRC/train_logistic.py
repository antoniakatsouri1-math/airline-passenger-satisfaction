import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

os.makedirs('models', exist_ok=True)
os.makedirs('src/plots', exist_ok=True)

def train(X_train, y_train, X_val, y_val, feature_names):

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Logistic Regression - Validation AUC-ROC: {val_auc:.4f}")

    coef = dict(zip(feature_names, np.abs(model.coef_[0])))
    top20 = sorted(coef.items(), key=lambda x: x[1], reverse=True)[:20]
    names, values = zip(*top20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names[::-1], values[::-1], color='coral')
    ax.set_title('Logistic Regression - Top 20 Coefficients (abs)')
    ax.set_xlabel('|Coefficient|')
    plt.tight_layout()
    plt.savefig('src/plots/lr_coefficients.png')
    plt.close()

    with open('models/logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Logistic Regression saved ✓")
    return model

if __name__ == '__main__':
    from SRC import preprocessing
    (X_train, X_val, X_test,
     y_train, y_val, y_test, feature_names) = preprocessing.run()
    train(X_train, y_train, X_val, y_val, feature_names)