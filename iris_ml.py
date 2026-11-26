"""
Iris flowers - machine learning with REAL data (beginner level)
Goal: guess the flower type from its petal and sepal size.
Run:  python iris_ml.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# STEP 1: load the real data into a pandas table
iris = load_iris()
data = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width",
                                        "petal_length", "petal_width"])
data["flower"] = [iris.target_names[i] for i in iris.target]
print(data.head(), "\n")
print(data["flower"].value_counts(), "\n")

# STEP 2: look at the data (each colour is one flower type)
sns.pairplot(data, hue="flower")
plt.savefig("iris_1_pairplot.png", dpi=100, bbox_inches="tight")
plt.show()

# STEP 3: split into inputs (X), answer (y), then train and test parts
X = data.drop("flower", axis=1)
y = data["flower"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# STEP 4: train the 3 models and check accuracy
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: accuracy = {results[name]:.2%}")

# STEP 5: confusion matrix of the best model
best_name = max(results, key=results.get)
best = models[best_name]
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best.predict(X_test), labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Model guess")
plt.ylabel("Real flower")
plt.title(f"Confusion matrix - {best_name}")
plt.savefig("iris_2_confusion.png", dpi=100, bbox_inches="tight")
plt.show()

# STEP 6: predict a new flower you measured yourself
new_flower = pd.DataFrame([{"sepal_length": 5.9, "sepal_width": 3.0,
                            "petal_length": 5.1, "petal_width": 1.8}])
print("\nNew flower prediction:")
for name, model in models.items():
    print(f"  {name}: {model.predict(new_flower)[0]}")
