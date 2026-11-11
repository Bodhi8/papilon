#   Copyright 2024 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
