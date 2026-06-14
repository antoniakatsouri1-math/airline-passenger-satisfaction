import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs('plots', exist_ok=True)

def run():
    df = pd.read_csv('../dataset.csv')
    rating_cols = [col for col in df.columns
                   if col not in df.select_dtypes(include='object').columns
                   and '_is_applicable' not in col
                   and col != 'liked']

    with open('eda_analysis_results.txt', 'w') as f:

        f.write("=== Rating Statistics ===\n")
        f.write(str(df[rating_cols].describe().round(2)) + "\n")

        f.write("\n=== Satisfaction ανά Process ===\n")
        f.write(str(df.groupby('process')['liked'].mean().round(3)) + "\n")

        f.write("\n=== Cross-tab: Gender x Age Group ===\n")
        cross = pd.crosstab(df['gender'], df['age_group'],
                           values=df['liked'], aggfunc='mean').round(3)
        f.write(str(cross) + "\n")

        f.write("\n=== Satisfaction ανά Traveling Alone ===\n")
        f.write(str(df.groupby('traveling_alone')['liked'].mean().round(3)) + "\n")

        f.write("\n=== Business vs Leisure - Mean Ratings ===\n")
        bl = df[df['trip_purpose'].isin(['Business', 'Leisure'])]
        f.write(str(bl.groupby('trip_purpose')[rating_cols].mean().round(2).T) + "\n")

        f.write("\n=== Chi-Square Tests ===\n")
        cat_cols = ['process', 'flight_type', 'gender', 'age_group',
                    'trip_purpose', 'traveling_alone']
        for col in cat_cols:
            ct = pd.crosstab(df[col], df['liked'])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            f.write(f"{col}: chi2={chi2:.2f}, p={p:.4f} {'important' if p < 0.05 else 'not important'}\n")

        f.write("\n=== Lowest rated services ===\n")
        means = df[rating_cols].mean().sort_values()
        f.write(str(means.head(10).round(3)) + "\n")

        f.write("\n=== Domestic vs International - Mean Ratings ===\n")
        f.write(str(df.groupby('flight_type')[rating_cols].mean().round(2).T) + "\n")

    top_features = df[rating_cols + ['liked']].corr()['liked'].drop('liked')\
                     .abs().sort_values(ascending=False).head(15).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap - Top 15 Features Correlation')
    plt.tight_layout()
    plt.savefig('plots/heatmap_top_features.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    df.groupby('process')['liked'].mean().plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.set_title('Satisfaction Rate / Process')
    ax.set_ylabel('% Liked')
    plt.tight_layout()
    plt.savefig('plots/liked_by_process.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    df[rating_cols].mean().sort_values().head(10).plot(kind='barh', ax=ax, color='tomato')
    ax.set_title('10 Lowest Rated Services')
    ax.set_xlabel('Mean Rating')
    plt.tight_layout()
    plt.savefig('plots/lowest_rated_services.png')
    plt.close()

    bl = df[df['trip_purpose'].isin(['Business', 'Leisure'])]
    top10 = df[rating_cols + ['liked']].corr()['liked'].drop('liked')\
              .abs().sort_values(ascending=False).head(10).index.tolist()
    means_bl = bl.groupby('trip_purpose')[top10].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    means_bl.T.plot(kind='bar', ax=ax)
    ax.set_title('Business vs Leisure - Top 10 Features')
    ax.set_ylabel('Mean Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/business_vs_leisure.png')
    plt.close()

    print("Analysis & plots saved")