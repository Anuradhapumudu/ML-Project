"""
================================================================================
  ML RETAIL PROJECT — Customer Transaction Value Prediction
  Dataset  : UCI Online Retail Dataset (synthetic fallback if unavailable)
  Models   : Logistic Regression | Decision Tree | K-Nearest Neighbors
  Author   : Anuradha Pumudu
  Run with : python ml_retail_project.py
================================================================================
"""

# === AUTO-INSTALL MISSING LIBRARIES ===
import subprocess
import sys

REQUIRED = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "requests", "openpyxl"]

def install_if_missing(packages):
    for pkg in packages:
        import_name = pkg.replace("-", "_").split(">=")[0]
        try:
            __import__(import_name)
        except ImportError:
            print(f"[SETUP] Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

install_if_missing(REQUIRED)

# === STANDARD IMPORTS ===
import os
import warnings
import requests
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, f1_score)

warnings.filterwarnings("ignore")

# ── global style ──────────────────────────────────────────────────────────────
PALETTE  = ["#6C63FF", "#FF6584", "#43B89C"]
BG_COLOR = "#0F0F1A"
FG_COLOR = "#E8E8F0"
sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : "#1A1A2E",
    "axes.edgecolor"   : "#333355",
    "axes.labelcolor"  : FG_COLOR,
    "xtick.color"      : FG_COLOR,
    "ytick.color"      : FG_COLOR,
    "text.color"       : FG_COLOR,
    "grid.color"       : "#2A2A4A",
    "grid.linewidth"   : 0.6,
    "font.family"      : "DejaVu Sans",
})

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# =============================================================================
# === SECTION 1: DATA LOADING =================================================
# =============================================================================

def download_dataset(save_path="online_retail.xlsx"):
    """Download the UCI Online Retail dataset (Excel format)."""
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00352/Online%20Retail.xlsx")
    print(f"[DATA] Downloading Online Retail dataset from UCI …")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[DATA] Saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"[WARN] Download failed ({e}). Generating synthetic dataset …")
        return None


def generate_synthetic_dataset(n=50_000, seed=42):
    """
    Create a realistic synthetic Online Retail dataset.
    Mirrors the schema of the original UCI dataset.
    """
    rng = np.random.default_rng(seed)
    countries = [
        "United Kingdom","Germany","France","EIRE","Spain","Netherlands",
        "Belgium","Switzerland","Portugal","Australia","Norway","Italy",
        "Channel Islands","Finland","Cyprus","Sweden","Austria","Denmark",
        "Japan","Poland","Israel","USA","Hong Kong","Singapore","Iceland",
        "Canada","Greece","Malta","United Arab Emirates","Brazil","Lebanon",
        "Lithuania","Japan","RSA","Bahrain","Czech Republic","Nigeria",
    ]
    n_countries = len(countries)
    qty      = rng.integers(1, 200, size=n)
    price    = np.round(rng.uniform(0.10, 50.0, size=n), 2)
    country  = rng.choice(countries, size=n,
                          p=np.array([30]+[2]*(n_countries-1), dtype=float) /
                            (30 + 2*(n_countries-1)))
    cust_id  = rng.integers(10000, 18500, size=n).astype(float)

    df = pd.DataFrame({
        "InvoiceNo"  : [f"INV{i:06d}" for i in range(n)],
        "StockCode"  : rng.integers(10000, 99999, size=n).astype(str),
        "Description": rng.choice(["WIDGET","GADGET","TOOL","ITEM","PART"], size=n),
        "Quantity"   : qty,
        "InvoiceDate": pd.date_range("2010-12-01", periods=n, freq="1min"),
        "UnitPrice"  : price,
        "CustomerID" : cust_id,
        "Country"    : country,
    })
    # inject some messy rows to make preprocessing realistic
    bad_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[bad_idx[:len(bad_idx)//2], "CustomerID"] = np.nan
    df.loc[bad_idx[len(bad_idx)//2:], "Quantity"]   = -rng.integers(1, 20, size=len(bad_idx)//2)
    return df


def load_dataset():
    """Load dataset: try UCI download → local file → synthetic fallback."""
    local_xlsx = "online_retail.xlsx"
    local_csv  = "online_retail.csv"

    # 1) try existing local xlsx
    if os.path.exists(local_xlsx):
        print(f"[DATA] Loading existing file: {local_xlsx}")
        try:
            return pd.read_excel(local_xlsx, engine="openpyxl")
        except Exception as e:
            print(f"[WARN] Could not read Excel ({e})")

    # 2) try existing local csv
    if os.path.exists(local_csv):
        print(f"[DATA] Loading existing file: {local_csv}")
        return pd.read_csv(local_csv, encoding="ISO-8859-1")

    # 3) try downloading
    path = download_dataset(local_xlsx)
    if path and os.path.exists(path):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            print(f"[WARN] Could not parse downloaded file ({e})")

    # 4) synthetic fallback
    print("[DATA] Using synthetic dataset (50,000 rows).")
    return generate_synthetic_dataset()


# =============================================================================
# === SECTION 2: PREPROCESSING ================================================
# =============================================================================

def preprocess(df):
    """Clean data, engineer target, encode and scale features."""
    print(f"\n[PREPROCESS] Raw shape: {df.shape}")

    # ── 2a. drop missing CustomerID ──────────────────────────────────────────
    df = df.dropna(subset=["CustomerID"])
    print(f"[PREPROCESS] After dropping null CustomerID: {df.shape}")

    # ── 2b. remove negatives (returns / errors) ───────────────────────────────
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"[PREPROCESS] After removing negatives: {df.shape}")

    # ── 2c. feature engineering ───────────────────────────────────────────────
    df = df.copy()
    df["Revenue"]   = df["Quantity"] * df["UnitPrice"]
    median_rev      = df["Revenue"].median()
    df["HighValue"] = (df["Revenue"] > median_rev).astype(int)
    print(f"[PREPROCESS] Revenue median = £{median_rev:.2f}")

    # ── 2d. encode Country ────────────────────────────────────────────────────
    le = LabelEncoder()
    df["Country_encoded"] = le.fit_transform(df["Country"].astype(str))

    # ── 2e. select features & target ─────────────────────────────────────────
    FEATURES = ["Quantity", "UnitPrice", "Country_encoded"]
    TARGET   = "HighValue"

    X = df[FEATURES].values
    y = df[TARGET].values

    # ── 2f. scale features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2g. train/test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[PREPROCESS] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[PREPROCESS] Class balance — HighValue=1: "
          f"{y.sum()} ({y.mean()*100:.1f}%), HighValue=0: "
          f"{(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    return df, X_train, X_test, y_train, y_test, FEATURES


# =============================================================================
# === SECTION 3: KNN ELBOW METHOD =============================================
# =============================================================================

def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """Run 5-Fold CV for each k and return (best_k, cv_scores_list)."""
    print("\n[KNN] Running Elbow Method (k = 1 to 20) …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for k in k_range:
        knn   = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1).mean()
        cv_scores.append(score)
        print(f"  k={k:2d}  CV Accuracy = {score:.4f}")

    best_k = list(k_range)[int(np.argmax(cv_scores))]
    print(f"[KNN] Best k = {best_k}  (CV Acc = {max(cv_scores):.4f})")
    return best_k, cv_scores


# =============================================================================
# === SECTION 4: MODEL TRAINING & EVALUATION ==================================
# =============================================================================

def train_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train a model and return a results dict with:
      test_acc, cv_acc, report, cm, best_f1
    """
    print(f"\n[MODEL] Training: {name} …")

    # train
    model.fit(X_train, y_train)

    # predictions
    y_pred = model.predict(X_test)

    # metrics
    test_acc = accuracy_score(y_test, y_pred)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc   = cross_val_score(model, X_train, y_train, cv=cv,
                               scoring="accuracy", n_jobs=-1).mean()
    report   = classification_report(y_test, y_pred, target_names=["Low Value","High Value"])
    cm       = confusion_matrix(y_test, y_pred)
    best_f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  CV  Accuracy  : {cv_acc:.4f}")
    print(f"  Weighted F1   : {best_f1:.4f}")
    print(f"\n{report}")

    return {
        "name"     : name,
        "model"    : model,
        "test_acc" : test_acc,
        "cv_acc"   : cv_acc,
        "report"   : report,
        "cm"       : cm,
        "best_f1"  : best_f1,
        "y_pred"   : y_pred,
    }


# =============================================================================
# === SECTION 5: VISUALIZATIONS ===============================================
# =============================================================================

def save_fig(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[PLOT] Saved → {path}")


def plot_customer_value_distribution(df):
    """Plot 1 — Customer Value Distribution (countplot of HighValue)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts  = df["HighValue"].value_counts().sort_index()
    bars    = ax.bar(["Low Value\n(0)", "High Value\n(1)"],
                     counts.values,
                     color=[PALETTE[1], PALETTE[0]],
                     edgecolor="#333355", linewidth=1.2, width=0.5)

    # annotate bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                f"{val:,}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, color=FG_COLOR, fontweight="bold")

    ax.set_title("Customer Transaction Value Distribution",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    ax.set_xlabel("Transaction Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig("01_customer_value_distribution.png")


def plot_top10_countries(df):
    """Plot 2 — Top 10 Countries by Revenue (horizontal bar)."""
    top10 = (df.groupby("Country")["Revenue"]
               .sum()
               .sort_values(ascending=False)
               .head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = sns.color_palette("viridis", n_colors=10)[::-1]
    bars    = ax.barh(top10.index[::-1], top10.values[::-1],
                      color=colors, edgecolor="#333355", linewidth=0.8)

    for bar, val in zip(bars, top10.values[::-1]):
        ax.text(bar.get_width() + top10.max()*0.005, bar.get_y() + bar.get_height()/2,
                f"£{val:,.0f}", va="center", fontsize=9, color=FG_COLOR)

    ax.set_title("Top 10 Countries by Total Revenue",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    ax.set_xlabel("Total Revenue (£)", fontsize=11)
    ax.set_ylabel("Country", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig("02_top10_countries_revenue.png")


def plot_correlation_heatmap(df):
    """Plot 3 — Feature Correlation Heatmap."""
    cols = ["Quantity", "UnitPrice", "Country_encoded", "Revenue", "HighValue"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask    = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, linecolor="#333355",
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 11, "color": "white"})

    ax.set_title("Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    plt.tight_layout()
    save_fig("03_feature_correlation_heatmap.png")


def plot_knn_elbow(k_range, cv_scores, best_k):
    """Plot 4 — KNN Elbow Curve (k vs CV Accuracy)."""
    ks = list(k_range)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ks, cv_scores, color=PALETTE[0], linewidth=2.5,
            marker="o", markersize=7, markerfacecolor=PALETTE[2],
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    # highlight best k
    best_score = cv_scores[ks.index(best_k)]
    ax.scatter([best_k], [best_score], color=PALETTE[1], s=180, zorder=5,
               label=f"Best k = {best_k}  (CV Acc = {best_score:.4f})")
    ax.axvline(best_k, color=PALETTE[1], linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_title("KNN Elbow Method — Cross-Validation Accuracy vs k",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    ax.set_xlabel("Number of Neighbours (k)", fontsize=11)
    ax.set_ylabel("5-Fold CV Accuracy", fontsize=11)
    ax.set_xticks(ks)
    ax.legend(fontsize=10, facecolor="#1A1A2E", edgecolor="#6C63FF", labelcolor=FG_COLOR)
    plt.tight_layout()
    save_fig("04_knn_elbow_curve.png")


def plot_confusion_matrices(results):
    """Plot 5 — Confusion Matrices for all 3 models side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))

    for ax, res, color in zip(axes, results, PALETTE):
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.5, linecolor="#333355",
                    xticklabels=["Low","High"],
                    yticklabels=["Low","High"],
                    ax=ax, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(res["name"], fontsize=12, fontweight="bold",
                     color=color, pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — All Models",
                 fontsize=15, fontweight="bold", color=FG_COLOR, y=1.02)
    plt.tight_layout()
    save_fig("05_confusion_matrices.png")


def plot_model_comparison(results):
    """Plot 6 — Grouped bar chart: Test Accuracy vs CV Accuracy per model."""
    names     = [r["name"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    cv_accs   = [r["cv_acc"]   for r in results]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, test_accs, width, label="Test Accuracy",
                   color=PALETTE[0], edgecolor="#333355", linewidth=0.8)
    bars2 = ax.bar(x + width/2, cv_accs,   width, label="CV  Accuracy (5-Fold)",
                   color=PALETTE[2], edgecolor="#333355", linewidth=0.8)

    # annotate
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=9.5,
                    color=FG_COLOR, fontweight="bold")

    ax.set_title("Model Performance Comparison",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10, facecolor="#1A1A2E",
              edgecolor="#6C63FF", labelcolor=FG_COLOR)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig("06_model_comparison.png")


def plot_feature_importance(dt_result, feature_names):
    """Plot 7 — Decision Tree Feature Importance."""
    importances = dt_result["model"].feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [PALETTE[0], PALETTE[2], PALETTE[1]]
    bars    = ax.bar(sorted_feats, sorted_vals,
                     color=[colors[i % len(colors)] for i in range(len(sorted_feats))],
                     edgecolor="#333355", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=11, color=FG_COLOR, fontweight="bold")

    ax.set_title("Decision Tree — Feature Importance",
                 fontsize=14, fontweight="bold", pad=15, color=FG_COLOR)
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Importance Score", fontsize=11)
    ax.set_ylim(0, max(sorted_vals) * 1.18)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    save_fig("07_decision_tree_feature_importance.png")


# =============================================================================
# === SECTION 6: SUMMARY TABLE ================================================
# =============================================================================

def print_summary(results):
    """Print a formatted model comparison table and highlight the winner."""
    header = f"\n{'='*68}\n{'MODEL PERFORMANCE SUMMARY':^68}\n{'='*68}"
    print(header)
    print(f"{'Model':<24} {'Test Acc':>10} {'CV Acc':>10} {'Best F1':>10}")
    print("-" * 68)

    best_test = max(results, key=lambda r: r["test_acc"])
    for r in results:
        marker = "  ← BEST" if r["name"] == best_test["name"] else ""
        print(f"{r['name']:<24} "
              f"{r['test_acc']:>10.4f} "
              f"{r['cv_acc']:>10.4f} "
              f"{r['best_f1']:>10.4f}"
              f"{marker}")
    print("=" * 68)
    print(f"\n🏆  BEST MODEL: {best_test['name'].upper()}")
    print(f"    Test Accuracy : {best_test['test_acc']:.4f}")
    print(f"    CV  Accuracy  : {best_test['cv_acc']:.4f}")
    print(f"    Weighted F1   : {best_test['best_f1']:.4f}")


# =============================================================================
# === MAIN ====================================================================
# =============================================================================

def main():
    print("\n" + "="*68)
    print("  ML RETAIL PROJECT — Customer Transaction Value Prediction")
    print("="*68)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df_raw = load_dataset()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    df, X_train, X_test, y_train, y_test, features = preprocess(df_raw)

    # ── 3. Visualisations (data exploration) ──────────────────────────────────
    print("\n[PLOTS] Generating exploratory visualisations …")
    plot_customer_value_distribution(df)
    plot_top10_countries(df)
    plot_correlation_heatmap(df)

    # ── 4. KNN Elbow Method ───────────────────────────────────────────────────
    K_RANGE   = range(1, 21)
    best_k, knn_cv_scores = find_best_k(X_train, y_train, K_RANGE)
    plot_knn_elbow(K_RANGE, knn_cv_scores, best_k)

    # ── 5. Define models ──────────────────────────────────────────────────────
    models = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("KNN (k={})".format(best_k),
         KNeighborsClassifier(n_neighbors=best_k)),
    ]

    # ── 6. Train & evaluate all models ───────────────────────────────────────
    results = []
    for name, model in models:
        res = train_evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # ── 7. Confusion matrices (all models) ───────────────────────────────────
    plot_confusion_matrices(results)

    # ── 8. Model comparison chart ────────────────────────────────────────────
    plot_model_comparison(results)

    # ── 9. Decision Tree feature importance ──────────────────────────────────
    dt_result = next(r for r in results if "Decision Tree" in r["name"])
    plot_feature_importance(dt_result, features)

    # ── 10. Final summary ────────────────────────────────────────────────────
    print_summary(results)

    print("\n" + "="*68)
    print("  ✅  PROJECT COMPLETE — All plots saved to /plots")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()
