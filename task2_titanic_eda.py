"""Task 2: exploratory data analysis on the Titanic dataset."""
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types and non-null counts:")
df.info()
print("\nMissing values by column:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nNumeric summary:")
print(df.describe(numeric_only=True))

# Use a copy so the raw source remains unchanged.
clean = df.copy()
for col in clean.select_dtypes(include="number").columns:
    clean[col] = clean[col].fillna(clean[col].median())
for col in clean.select_dtypes(exclude="number").columns:
    if clean[col].isna().any():
        clean[col] = clean[col].fillna(clean[col].mode(dropna=True).iloc[0])

print("\nSurvival rate by passenger class:")
print(clean.groupby("pclass", observed=False)["survived"].mean().round(3))
print("\nSurvival rate by sex:")
print(clean.groupby("sex", observed=False)["survived"].mean().round(3))
print("\nCleaned missing values remaining:", int(clean.isna().sum().sum()))
