import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)

def run():
    df = pd.read_csv('data/dataset.csv')

    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby('age_group')['liked'].mean().sort_values().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Satisfaction Rate ανά Age Group')
    ax.set_ylabel('% Liked')
    ax.set_xlabel('Age Group')
    plt.tight_layout()
    plt.savefig('plots/liked_by_age_group.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    df.groupby('trip_purpose')['liked'].mean().sort_values().plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Satisfaction Rate ανά Trip Purpose')
    ax.set_ylabel('% Liked')
    ax.set_xlabel('Trip Purpose')
    plt.tight_layout()
    plt.savefig('plots/liked_by_trip_purpose.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    df.groupby('flight_type')['liked'].mean().plot(kind='bar', ax=ax, color='mediumseagreen')
    ax.set_title('Satisfaction Rate ανά Flight Type')
    ax.set_ylabel('% Liked')
    ax.set_xlabel('Flight Type')
    plt.tight_layout()
    plt.savefig('plots/liked_by_flight_type.png')
    plt.close()

    rating_cols = [col for col in df.columns
                   if col not in df.select_dtypes(include='object').columns
                   and '_is_applicable' not in col
                   and col != 'liked']

    corr = df[rating_cols + ['liked']].corr()['liked'].drop('liked')
    top10 = corr.abs().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    top10.sort_values().plot(kind='barh', ax=ax, color='mediumpurple')
    ax.set_title('Top 10 Features - Correlation με Liked')
    ax.set_xlabel('Correlation')
    plt.tight_layout()
    plt.savefig('plots/top10_correlations.png')
    plt.close()

    print("Plots saved in plots/")