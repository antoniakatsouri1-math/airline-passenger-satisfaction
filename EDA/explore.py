import pandas as pd

def run():
    df = pd.read_csv('data/dataset.csv')

    with open('explore_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== Shape ===\n")
        f.write(str(df.shape) + "\n\n")

        f.write("=== Columns ===\n")
        f.write(str(df.columns.tolist()) + "\n\n")

        f.write("=== Head(10) ===\n")
        f.write(str(df.head(10)) + "\n\n")

        f.write("=== Dtypes ===\n")
        f.write(str(df.dtypes) + "\n\n")

        f.write("=== Null Values ===\n")
        f.write(str(df.isnull().sum()) + "\n")

    print("explore_results.txt saved")