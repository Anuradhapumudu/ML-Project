# Online Retail ML Project
# Using the UCI Online Retail dataset to classify high/low value transactions
# Models: Logistic Regression, Decision Tree, KNN
# Student: Anuradha Pumudu

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

# make sure plots folder exists
if not os.path.exists("plots"):
    os.makedirs("plots")


# -------------------------------------------------------
# Step 1 - Load the dataset
# -------------------------------------------------------

# I tried downloading from UCI but the server is slow so i saved a local copy
# if local file not found, we generate synthetic data with same structure

def make_synthetic_data():
    # create fake retail data that looks like the real UCI dataset
    # this has about 50000 rows
    print("Generating synthetic retail dataset...")
    
    np.random.seed(42)
    n = 50000
    
    countries = [
        "United Kingdom", "Germany", "France", "EIRE", "Spain",
        "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia",
        "Norway", "Italy", "Finland", "Sweden", "Denmark",
        "Japan", "Poland", "USA", "Hong Kong", "Canada"
    ]
    
    # UK is the main market so give it higher probability
    probs = [0.5] + [0.5 / (len(countries) - 1)] * (len(countries) - 1)
    
    qty = np.random.randint(1, 150, size=n)
    price = np.round(np.random.uniform(0.5, 45.0, size=n), 2)
    country_col = np.random.choice(countries, size=n, p=probs)
    customer_ids = np.random.randint(10000, 18000, size=n).astype(float)
    
    df = pd.DataFrame({
        "InvoiceNo": ["INV" + str(i).zfill(6) for i in range(n)],
        "StockCode": np.random.randint(10000, 99999, size=n).astype(str),
        "Description": np.random.choice(["MUG", "BAG", "CANDLE", "FRAME", "TOY"], size=n),
        "Quantity": qty,
        "InvoiceDate": pd.date_range("2010-12-01", periods=n, freq="1min"),
        "UnitPrice": price,
        "CustomerID": customer_ids,
        "Country": country_col
    })
    
    # add some missing values and returns like the real dataset has
    missing_idx = np.random.choice(n, size=1500, replace=False)
    df.loc[missing_idx, "CustomerID"] = np.nan
    
    return_idx = np.random.choice(n, size=800, replace=False)
    df.loc[return_idx, "Quantity"] = -np.random.randint(1, 10, size=800)
    
    return df


def load_data():
    # try loading local excel file first
    if os.path.exists("online_retail.xlsx"):
        print("Loading dataset from local file...")
        try:
            df = pd.read_excel("online_retail.xlsx", engine="openpyxl")
            print("Dataset loaded successfully!")
            return df
        except Exception as e:
            print("Could not read excel file:", e)
    
    # try csv version
    if os.path.exists("online_retail.csv"):
        print("Loading from CSV...")
        df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
        return df
    
    # fallback to synthetic data
    print("Local dataset not found - using synthetic data instead")
    return make_synthetic_data()


print("=" * 60)
print("        Online Retail - Machine Learning Project")
print("=" * 60)

df = load_data()
print("\nDataset shape:", df.shape)
print(df.head())


# -------------------------------------------------------
# Step 2 - Data Preprocessing
# -------------------------------------------------------

print("\n--- Preprocessing ---")

# drop rows where CustomerID is missing
print("Removing rows with missing CustomerID...")
df = df.dropna(subset=["CustomerID"])
print("Shape after removing nulls:", df.shape)

# remove negative quantities (these are returns/cancellations)
# also remove zero or negative prices
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print("Shape after removing returns and bad prices:", df.shape)

# create Revenue column
df = df.copy()
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# create the target variable - HighValue is 1 if revenue above median
median_revenue = df["Revenue"].median()
print("Median revenue:", round(median_revenue, 2))

df["HighValue"] = (df["Revenue"] > median_revenue).astype(int)

print("Class distribution:")
print(df["HighValue"].value_counts())

# encode the Country column since its categorical
le = LabelEncoder()
df["Country_encoded"] = le.fit_transform(df["Country"].astype(str))

# select features for the model
features = ["Quantity", "UnitPrice", "Country_encoded"]
target = "HighValue"

X = df[features].values
y = df[target].values

# scale the features - important for KNN and Logistic Regression
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)


# -------------------------------------------------------
# Step 3 - Visualizations (Exploratory Data Analysis)
# -------------------------------------------------------

print("\n--- Creating Visualizations ---")

# Plot 1: Customer value distribution
plt.figure(figsize=(8, 5))
counts = df["HighValue"].value_counts().sort_index()
bars = plt.bar(["Low Value", "High Value"], counts.values, color=["#e74c3c", "#3498db"], width=0.5, edgecolor="black")

for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 100,
             str(val),
             ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.title("Customer Transaction Value Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("plots/01_customer_value_distribution.png", dpi=150)
plt.close()
print("Saved: customer value distribution")


# Plot 2: Top 10 countries by revenue
country_revenue = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
country_revenue[::-1].plot(kind="barh", color="steelblue", edgecolor="black")
plt.title("Top 10 Countries by Total Revenue", fontsize=14, fontweight="bold")
plt.xlabel("Total Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("plots/02_top10_countries_revenue.png", dpi=150)
plt.close()
print("Saved: top 10 countries chart")


# Plot 3: Correlation heatmap
corr_cols = ["Quantity", "UnitPrice", "Country_encoded", "Revenue", "HighValue"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_feature_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation heatmap")


# -------------------------------------------------------
# Step 4 - KNN Elbow Method (finding best k)
# -------------------------------------------------------

print("\n--- KNN Elbow Method ---")
print("Testing k values from 1 to 20...")

k_values = range(1, 21)
cv_scores_knn = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    avg_score = scores.mean()
    cv_scores_knn.append(avg_score)
    print("k =", k, " -> CV Accuracy:", round(avg_score, 4))

# find which k gave the best accuracy
best_k = list(k_values)[cv_scores_knn.index(max(cv_scores_knn))]
print("\nBest k:", best_k, "with CV accuracy:", round(max(cv_scores_knn), 4))


# Plot 4: KNN Elbow curve
plt.figure(figsize=(9, 5))
plt.plot(list(k_values), cv_scores_knn, marker="o", color="royalblue", linewidth=2, markersize=6)
plt.axvline(x=best_k, color="red", linestyle="--", label="Best k = " + str(best_k))
plt.scatter([best_k], [max(cv_scores_knn)], color="red", s=100, zorder=5)
plt.title("KNN - Elbow Method (k vs CV Accuracy)", fontsize=14, fontweight="bold")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("5-Fold CV Accuracy")
plt.xticks(list(k_values))
plt.legend()
plt.tight_layout()
plt.savefig("plots/04_knn_elbow_curve.png", dpi=150)
plt.close()
print("Saved: KNN elbow curve")


# -------------------------------------------------------
# Step 5 - Train and Evaluate Models
# -------------------------------------------------------

print("\n--- Training Models ---")

# helper function to train and evaluate a single model
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    print("\nTraining:", name)
    
    # fit the model
    model.fit(X_train, y_train)
    
    # make predictions on test set
    y_pred = model.predict(X_test)
    
    # calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Low Value", "High Value"])
    
    print("Test Accuracy:", round(test_accuracy, 4))
    print("CV Accuracy (5-fold):", round(cv_accuracy, 4))
    print("Weighted F1 Score:", round(f1, 4))
    print("\nClassification Report:")
    print(report)
    
    return {
        "name": name,
        "model": model,
        "test_acc": test_accuracy,
        "cv_acc": cv_accuracy,
        "f1": f1,
        "cm": cm,
        "y_pred": y_pred
    }


# Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_result = train_and_evaluate("Logistic Regression", lr_model, X_train, X_test, y_train, y_test)

# Model 2: Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_result = train_and_evaluate("Decision Tree", dt_model, X_train, X_test, y_train, y_test)

# Model 3: KNN with best k from elbow method
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_result = train_and_evaluate("KNN (k=" + str(best_k) + ")", knn_model, X_train, X_test, y_train, y_test)

# store all results in a list
all_results = [lr_result, dt_result, knn_result]


# -------------------------------------------------------
# Step 6 - Plot Confusion Matrices for all models
# -------------------------------------------------------

# Plot 5: all 3 confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ["Blues", "Greens", "Purples"]

for i, (result, color) in enumerate(zip(all_results, colors)):
    sns.heatmap(result["cm"],
                annot=True,
                fmt="d",
                cmap=color,
                ax=axes[i],
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"],
                linewidths=0.5)
    axes[i].set_title(result["name"], fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/05_confusion_matrices.png", dpi=150)
plt.close()
print("\nSaved: confusion matrices")


# -------------------------------------------------------
# Step 7 - Model Comparison Chart
# -------------------------------------------------------

# Plot 6: bar chart comparing test accuracy vs cv accuracy
model_names = [r["name"] for r in all_results]
test_accs = [r["test_acc"] for r in all_results]
cv_accs = [r["cv_acc"] for r in all_results]

x = np.arange(len(model_names))
bar_width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - bar_width / 2, test_accs, bar_width, label="Test Accuracy", color="#3498db", edgecolor="black")
bars2 = plt.bar(x + bar_width / 2, cv_accs, bar_width, label="CV Accuracy (5-fold)", color="#2ecc71", edgecolor="black")

# add value labels on the bars
for bar in bars1:
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.003,
             str(round(bar.get_height(), 4)),
             ha="center", va="bottom", fontsize=9)

for bar in bars2:
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.003,
             str(round(bar.get_height(), 4)),
             ha="center", va="bottom", fontsize=9)

plt.title("Model Comparison - Test Accuracy vs CV Accuracy", fontsize=13, fontweight="bold")
plt.xticks(x, model_names)
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("plots/06_model_comparison.png", dpi=150)
plt.close()
print("Saved: model comparison chart")


# Plot 7: Decision Tree feature importance
feature_importance = dt_model.feature_importances_
feature_labels = features

plt.figure(figsize=(8, 5))
plt.bar(feature_labels, feature_importance, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")

for i, val in enumerate(feature_importance):
    plt.text(i, val + 0.005, str(round(val, 4)), ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.title("Decision Tree - Feature Importance", fontsize=13, fontweight="bold")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/07_decision_tree_feature_importance.png", dpi=150)
plt.close()
print("Saved: feature importance chart")


# -------------------------------------------------------
# Step 8 - Final Summary Table
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("           FINAL MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Test Acc':>10} {'CV Acc':>10} {'F1 Score':>10}")
print("-" * 60)

best_model = max(all_results, key=lambda r: r["test_acc"])

for result in all_results:
    marker = " <-- BEST" if result["name"] == best_model["name"] else ""
    print(f"{result['name']:<25} {result['test_acc']:>10.4f} {result['cv_acc']:>10.4f} {result['f1']:>10.4f}{marker}")

print("=" * 60)
print("\nBest performing model:", best_model["name"])
print("Test Accuracy:", round(best_model["test_acc"], 4))
print("CV Accuracy:", round(best_model["cv_acc"], 4))
print("F1 Score:", round(best_model["f1"], 4))

print("\n--- PROJECT COMPLETE ---")
print("All plots saved in the /plots folder")
