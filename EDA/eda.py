import pandas as pd


def run():
    df = pd.read_csv('../dataset.csv')

    with open('eda_results.txt', 'w') as f:
        f.write("=== LIKED distribution ===\n")
        f.write(str(df['liked'].value_counts()) + "\n")
        f.write(str(df['liked'].value_counts(normalize=True).round(2)) + "\n")

        f.write("\n=== Categorical columns ===\n")
        cat_cols = ['process', 'flight_type', 'connection', 'ticket_purchased_by',
                    'gender', 'age_group', 'trip_purpose', 'traveling_alone']

        for col in cat_cols:
            f.write(f"\n{col}:\n")
            f.write(str(df[col].value_counts()) + "\n")

        rating_cols = [col for col in df.columns
                       if col not in df.select_dtypes(include='object').columns
                       and '_is_applicable' not in col
                       and col != 'liked']

        corr = df[rating_cols + ['liked']].corr()['liked'].drop('liked')
        corr_sorted = corr.abs().sort_values(ascending=False)

        f.write("\n=== Correlation with liked ===\n")
        f.write(str(corr_sorted.head(20)) + "\n")

    print("EDA saved in eda_results.txt")