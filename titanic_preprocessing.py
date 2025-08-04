import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("\nInitial Data Exploration")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst 5 Rows:")
    print(df.head())
    return df
def handle_missing_values(df):
    print("\nHandling Missing Values")
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
        print("Dropped 'Cabin' column (too many missing values)")
        num_cols = ['Age', 'Fare']
    for col in num_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} with median: {median_val:.2f}")
        cat_cols = ['Embarked']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled {col} with mode: {mode_val}")
    
    return df
def encode_categorical_features(df):
    print("\nEncoding Categorical Features")
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        print("Label encoded 'Sex' (male=0, female=1)")
        cat_cols = ['Embarked', 'Pclass']
    cat_cols = [col for col in cat_cols if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"One-hot encoded columns: {cat_cols}")
    
    return df

def normalize_numerical_features(df):
    print("\nStandardizing Numerical Features")
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    num_cols = [col for col in num_cols if col in df.columns]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"Standardized columns: {num_cols}")
    return df
def handle_outliers(df):
    print("\nHandling Outliers")
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    num_cols = [col for col in num_cols if col in df.columns]
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[num_cols])
    plt.title('Boxplots BEFORE Outlier Removal')
    plt.show()
    before = len(df)
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    after = len(df)
    print(f"Removed {before - after} rows with outliers from {num_cols}")
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[num_cols])
    plt.title('Boxplots AFTER Outlier Removal')
    plt.show()
    
    return df
if __name__ == "__main__":
    df = load_and_explore_data('Titanic-Dataset.csv')
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = normalize_numerical_features(df)
    df = handle_outliers(df)
    irrelevant_cols = ['Name', 'Ticket']
    df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns])
    print("\n Final Cleaned Dataset")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 Rows:")
    print(df.head())
    df.to_csv('cleaned_titanic.csv', index=False)
    print("\nCleaned data saved to 'cleaned_titanic.csv'")