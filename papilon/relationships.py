import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

def analyze_relationships(df, target=None, visualize=True):
    results = {}
    numeric_df = df.select_dtypes(include=['number']).dropna()

    if visualize:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    results['correlation_matrix'] = numeric_df.corr()

    if target and target in df.columns:
        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]
        mi = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        results['mutual_information'] = mi_series

        if visualize:
            plt.figure(figsize=(8, 5))
            mi_series.plot(kind='bar')
            plt.title(f'Mutual Information with Target: {target}')
            plt.ylabel('MI Score')
            plt.show()

    return results
