import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run():
    df = pd.read_csv('dataset.csv')

    drop_cols = [col for col in df.columns if '_is_applicable' in col]
    df = df.drop(columns=drop_cols)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median())

    scaler = StandardScaler()
    cols_to_scale = [col for col in df.columns if col != 'liked']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    df.to_csv('dataset_clean.csv', index=False)

    with open('Models/preprocessing_results.txt', 'w') as f:
        f.write("=== Preprocessing Results ===\n")
        f.write(f"Original shape: (57514, 145)\n")
        f.write(f"Columns dropped (_is_applicable): {len(drop_cols)}\n")
        f.write(f"Final shape: {df.shape}\n")
        f.write(f"NaN values after preprocessing: {df.isnull().sum().sum()}\n")
        f.write(f"Columns remaining: {df.columns.tolist()}\n")

    print(f"Preprocessing done. Shape: {df.shape} ✓")
    return df