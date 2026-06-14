import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

os.makedirs('models', exist_ok=True)
os.makedirs('src/plots', exist_ok=True)

def train(X_train, y_train, X_val, y_val, feature_names):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Random Forest - Validation AUC-ROC: {val_auc:.4f}")

    feat_imp = dict(zip(feature_names, model.feature_importances_))
    top20 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20]
    names, values = zip(*top20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names[::-1], values[::-1], color='steelblue')
    ax.set_title('Random Forest - Top 20 Feature Importance')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('src/plots/rf_feature_importance.png')
    plt.close()

    with open('models/classical_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Random Forest saved ✓")
    return model

if __name__ == '__main__':
    from SRC import preprocessing
    (X_train, X_val, X_test,
     y_train, y_val, y_test, feature_names) = preprocessing.run()
    train(X_train, y_train, X_val, y_val, feature_names)