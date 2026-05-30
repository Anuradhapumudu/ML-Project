# 🛍️ ML Retail Project — Customer Value Prediction

A complete end-to-end **Machine Learning project** built on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail).  
The goal is to predict whether a transaction is **High Value** or **Low Value** using three supervised learning algorithms.

---

## 📌 Project Objective

Build, evaluate, and compare multiple ML models to classify retail transactions by customer value, demonstrating a full data science workflow from raw data to insights.

---

## 📂 Project Structure

```
ML-Project/
│
├── ml_retail_project.py   # Main script — run this end-to-end
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── plots/                 # Auto-generated visualizations (PNG)
    ├── 01_customer_value_distribution.png
    ├── 02_top10_countries_revenue.png
    ├── 03_feature_correlation_heatmap.png
    ├── 04_knn_elbow_curve.png
    ├── 05_confusion_matrices.png
    └── 06_model_comparison.png
    └── 07_decision_tree_feature_importance.png
```

---

## 🗃️ Dataset

**Online Retail Dataset** — UCI Machine Learning Repository  
- Transactions from a UK-based online retailer (2010–2011)  
- ~500,000 rows | Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### Features Used
| Feature | Description |
|---|---|
| `Quantity` | Number of units purchased |
| `UnitPrice` | Price per unit (GBP) |
| `Country` | Customer's country (label-encoded) |

### Target Variable
- **Revenue** = `Quantity × UnitPrice`  
- **HighValue** = `1` if Revenue > median, else `0`

---

## 🤖 Machine Learning Models

| Model | Key Parameters |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=5`, `random_state=42` |
| K-Nearest Neighbors | Best `k` via Elbow Method (k=1–20) |

---

## 📊 Evaluation Metrics

- **Test Accuracy**
- **5-Fold Cross-Validation Accuracy**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (Seaborn heatmap)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Anuradhapumudu/ML-Project.git
cd ML-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python ml_retail_project.py
```

> All missing libraries are auto-installed. Plots are saved to `/plots`.

---

## 📈 Visualizations

1. Customer Value Distribution (HighValue countplot)
2. Top 10 Countries by Revenue (horizontal bar)
3. Feature Correlation Heatmap
4. KNN Elbow Curve (k vs CV Accuracy)
5. Confusion Matrices — all 3 models side by side
6. Model Comparison Bar Chart (Test vs CV Accuracy)
7. Decision Tree Feature Importance

---

## 🧪 Results Summary

| Model | Test Accuracy | CV Accuracy | Best F1 |
|---|---|---|---|
| Logistic Regression | 0.9140 | 0.9114 | 0.9140 |
| Decision Tree | 0.9777 | 0.9820 | 0.9777 |
| **KNN (k=19)** ⭐ | **0.9927** | **0.9930** | **0.9927** |

> 🏆 **Best Model: KNN (k=19)** with **99.27% test accuracy** and **99.30% CV accuracy**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4c8cbf)

---

## 📚 Learning Outcomes

- Full ML workflow: data loading → preprocessing → training → evaluation
- Feature engineering and target variable creation
- Categorical encoding with `LabelEncoder`
- Feature scaling with `StandardScaler`
- Model comparison and selection

---

## 👤 Author

**Anuradha Pumudu**  
GitHub: [@Anuradhapumudu](https://github.com/Anuradhapumudu)
