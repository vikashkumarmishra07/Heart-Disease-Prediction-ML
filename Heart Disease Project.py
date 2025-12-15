#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Load Dataset
df = pd.read_csv(
    r"C:\Users\Vikas mishra\Desktop\INT234 PREDICTIVE ANALYSIS\archive\heart.csv"
)

print("\nDataset Loaded Successfully")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

#Data Cleaning

print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates()
print("\nDuplicates Removed")

#EDA & Visualization

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(x="target", data=df)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Outlier Detection
plt.figure(figsize=(12,4))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Outlier Detection")
plt.show()

# Feature Comparison
plt.figure(figsize=(8,5))
sns.histplot(df[df["target"] == 1]["age"], color="red", label="Disease", kde=True)
sns.histplot(df[df["target"] == 0]["age"], color="green", label="No Disease", kde=True)
plt.legend()
plt.title("Age Distribution by Target")
plt.show()

#Feature Selection

X = df.drop("target", axis=1)
y = df["target"]

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model Building
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True)
}

results = []

#Model Training & Evaluation
print("\nMODEL PERFORMANCE\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results.append([name, acc])

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

#Accuracy Comparison
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\nModel Accuracy Comparison")
print(results_df)

plt.figure(figsize=(8,5))
sns.barplot(x="Accuracy", y="Model", data=results_df)
plt.title("Model Accuracy Comparison")
plt.show()
