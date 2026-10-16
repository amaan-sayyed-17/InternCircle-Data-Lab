"""Task 3: create useful visualisations from Titanic data."""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = sns.load_dataset("titanic")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.histplot(data=df, x="age", hue="survived", bins=25, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age distribution by survival")
sns.countplot(data=df, x="class", hue="survived", ax=axes[0, 1])
axes[0, 1].set_title("Survival counts by passenger class")
sns.boxplot(data=df, x="survived", y="fare", ax=axes[1, 0])
axes[1, 0].set_title("Fare distribution by survival")
numeric = df.select_dtypes(include="number")
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title("Correlation heatmap")
plt.tight_layout()
plt.savefig("titanic_visualizations.png", dpi=150, bbox_inches="tight")
plt.show()
