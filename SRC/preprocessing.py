import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import os

os.makedirs('Models', exist_ok=True)
os.makedirs('SRC/plots', exist_ok=True)

def load_data():
    df = pd.read_csv('data/dataset.csv')
    df = df.replace('<null>', np.nan)

    if 'connection_wait_time' in df.columns:
        df['connection_wait_time'] = df['connection_wait_time'] / 60
    if 'arrival_lead_time' in df.columns:
        df['arrival_lead_time'] = df['arrival_lead_time'] / 60

    return df

def split_data(df):
    X = df.drop(columns=['liked'])
    y = df['liked']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def drop_unnecessary_cols(X_train, X_val, X_test):
    drop_cols = [col for col in X_train.columns if '_is_applicable' in col]
    drop_cols += ['flight_type', 'gender']
    drop_cols = [col for col in drop_cols if col in X_train.columns]

    return (X_train.drop(columns=drop_cols),
            X_val.drop(columns=drop_cols),
            X_test.drop(columns=drop_cols))

def impute_missing(X_train, X_val, X_test):
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_val[num_cols] = X_val[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)

    cat_cols = X_train.select_dtypes(include='object').columns
    modes = X_train[cat_cols].mode().iloc[0]
    X_train[cat_cols] = X_train[cat_cols].fillna(modes)
    X_val[cat_cols] = X_val[cat_cols].fillna(modes)
    X_test[cat_cols] = X_test[cat_cols].fillna(modes)

    return X_train, X_val, X_test

def treat_outliers(X_train, X_val, X_test):
    num_cols = [col for col in X_train.select_dtypes(include=['int64', 'float64']).columns
                if X_train[col].nunique() > 10]

    for col in num_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    return X_train, X_val, X_test

def encode_categoricals(X_train, X_val, X_test):
    cat_cols = X_train.select_dtypes(include='object').columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = X_val[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        X_test[col] = X_test[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        encoders[col] = le

    # Αποθήκευση encoders για χρήση στο API
    with open('Models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("Encoders saved ✓")

    return X_train, X_val, X_test

def feature_engineering(X_train, X_val, X_test):
    for dataset in [X_train, X_val, X_test]:
        comfort_cols = ['boarding_lounge_comfort', 'thermal_comfort', 'acoustic_comfort']
        dataset['comfort_score'] = dataset[comfort_cols].mean(axis=1)

        clean_cols = ['overall_airport_cleanliness', 'restroom_cleanliness', 'restroom_maintenance']
        dataset['cleanliness_score'] = dataset[clean_cols].mean(axis=1)

        price_cols = ['food_beverage_price_quality', 'retail_price_quality', 'parking_value_for_money']
        dataset['price_quality_score'] = dataset[price_cols].mean(axis=1)

        queue_cols = ['checkin_queue_wait_time', 'security_queue_wait_time']
        dataset['queue_score'] = dataset[queue_cols].mean(axis=1)

    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open('Models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('Models/feature_names.pkl', 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)

    # Αποθήκευση medians για missing features στο API
    medians = X_train.median().to_dict()
    with open('Models/medians.pkl', 'wb') as f:
        pickle.dump(medians, f)

    print("Scaler, feature names & medians saved ✓")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def run_pca(X_train_scaled, y_train):
    pca_full = PCA(random_state=42)
    pca_full.fit(X_train_scaled)

    explained = np.cumsum(pca_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(explained) + 1), explained, marker='o', markersize=3)
    ax.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
    ax.set_title('PCA - Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.legend()
    plt.tight_layout()
    plt.savefig('SRC/plots/pca_scree.png')
    plt.close()

    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_train_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train,
                         cmap='coolwarm', alpha=0.3, s=5)
    plt.colorbar(scatter, label='liked')
    ax.set_title('PCA - 2D Projection by Class')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig('SRC/plots/pca_2d.png')
    plt.close()

    n_90 = np.argmax(explained >= 0.90) + 1
    print(f"Components for 90% variance: {n_90}")

    return pca_full

def run():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = drop_unnecessary_cols(X_train, X_val, X_test)
    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)
    X_train, X_val, X_test = treat_outliers(X_train, X_val, X_test)
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test)
    X_train, X_val, X_test = feature_engineering(X_train, X_val, X_test)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test)
    run_pca(X_train_scaled, y_train)

    print("Preprocessing complete ✓")
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            X_train.columns.tolist())

if __name__ == '__main__':
    run()