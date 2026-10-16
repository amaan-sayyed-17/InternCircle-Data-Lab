"""Task 5: train and evaluate a Decision Tree classifier."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Features:", list(X.columns))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("\nNote: this is a baseline educational model; evaluate it before drawing conclusions.")
