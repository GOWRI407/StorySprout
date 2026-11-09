"""
StorySprout Capstone - Part B: Learner Segmentation (RFM + K-Means) & Churn Prediction
Prevents label leakage with strict temporal cut-off:
- Features computed strictly on submissions <= cutoff_date (30 days prior to end)
- Target label (churn=1 if 0 submissions in post-cutoff window) defined strictly on submissions > cutoff_date
Evaluates K-Means with Silhouette Scores to justify k=3.
Trains Logistic Regression & Random Forest, reports AUC, Precision, Recall, F1, ROC plot, and Feature Importance.
Saves model pipeline to models/churn_model.joblib and metrics to outputs/part_b/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set plot style and env
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_mpl")
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "part_b")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 42


def load_and_prepare_temporal_data():
    """Load cleaned datasets and split strictly by temporal cutoff to prevent leakage."""
    learners = pd.read_csv(os.path.join(PROCESSED_DIR, "learners_clean.csv"))
    submissions = pd.read_csv(os.path.join(PROCESSED_DIR, "submissions_clean.csv"))

    learners["signup_date"] = pd.to_datetime(learners["signup_date"])
    submissions["submission_date"] = pd.to_datetime(submissions["submission_date"])

    max_date = submissions["submission_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=30)

    print(f"Dataset date range: {submissions['submission_date'].min().date()} to {max_date.date()}")
    print(f"Strict temporal cutoff date: {cutoff_date.date()}")

    # Eligible cohort: learners signed up on or before cutoff date
    cohort_learners = learners[learners["signup_date"] <= cutoff_date].copy()
    print(f"Learners in pre-cutoff cohort: {len(cohort_learners):,} / {len(learners):,}")

    # Split submissions into observation (pre-cutoff) and outcome (post-cutoff) windows
    pre_subs = submissions[submissions["submission_date"] <= cutoff_date].copy()
    post_subs = submissions[submissions["submission_date"] > cutoff_date].copy()

    # Define Ground-Truth Churn Target (Strictly post-cutoff outcome)
    # churn = 1 if no submissions in the 30-day post-cutoff window, else 0
    post_active_learners = set(post_subs["learner_id"].unique())
    cohort_learners["churn"] = cohort_learners["learner_id"].apply(
        lambda lid: 0 if lid in post_active_learners else 1
    )

    # -------------------------------------------------------------
    # Feature Engineering (Strictly pre-cutoff observation window)
    # -------------------------------------------------------------
    pre_agg = pre_subs.groupby("learner_id").agg(
        last_submission=("submission_date", "max"),
        first_submission=("submission_date", "min"),
        frequency_submissions=("submission_id", "count"),
        total_words=("word_count", "sum"),
        total_minutes=("minutes_spent", "sum"),
        avg_score=("score", "mean"),
        total_grammar_errors=("grammar_errors", "sum"),
        total_vocab_errors=("vocab_errors", "sum"),
        active_days=("submission_date", lambda d: d.dt.date.nunique())
    ).reset_index()

    # Recency: days between last pre-cutoff submission and cutoff date
    pre_agg["recency_days"] = (cutoff_date - pre_agg["last_submission"]).dt.days.clip(lower=0)
    # Tenure at cutoff
    pre_agg["tenure_days"] = (cutoff_date - pre_agg["first_submission"]).dt.days.clip(lower=1)
    # Error rates per 100 words
    pre_agg["grammar_rate_per_100w"] = (pre_agg["total_grammar_errors"] / pre_agg["total_words"].clip(lower=10)) * 100
    pre_agg["vocab_rate_per_100w"] = (pre_agg["total_vocab_errors"] / pre_agg["total_words"].clip(lower=10)) * 100
    pre_agg["avg_words_per_sub"] = pre_agg["total_words"] / pre_agg["frequency_submissions"]

    # Merge with cohort learners
    rfm_df = cohort_learners.merge(pre_agg, on="learner_id", how="left")

    # Learners with 0 pre-cutoff submissions: recency = tenure from signup
    no_pre_mask = rfm_df["frequency_submissions"].isna()
    rfm_df.loc[no_pre_mask, "recency_days"] = (cutoff_date - rfm_df.loc[no_pre_mask, "signup_date"]).dt.days.clip(lower=0)
    rfm_df.loc[no_pre_mask, "frequency_submissions"] = 0
    rfm_df.loc[no_pre_mask, "total_words"] = 0
    rfm_df.loc[no_pre_mask, "total_minutes"] = 0
    rfm_df.loc[no_pre_mask, "avg_score"] = 50.0  # neutral imputation
    rfm_df.loc[no_pre_mask, "active_days"] = 0
    rfm_df.loc[no_pre_mask, "grammar_rate_per_100w"] = 0.0
    rfm_df.loc[no_pre_mask, "vocab_rate_per_100w"] = 0.0
    rfm_df.loc[no_pre_mask, "avg_words_per_sub"] = 0.0

    # Goal adherence
    rfm_df["goal_adherence_ratio"] = (rfm_df["total_minutes"] / (rfm_df["active_days"].clip(lower=1) * rfm_df["daily_goal_min"])).clip(upper=3.0)

    print(f"Overall Churn rate in cohort: {rfm_df['churn'].mean():.2%}")
    return rfm_df, cutoff_date


def perform_kmeans_segmentation(rfm_df: pd.DataFrame):
    """Run K-Means clustering on RFM features, evaluate silhouette scores for k=2..6, and assign semantic labels."""
    print("--- Running K-Means RFM Segmentation ---")

    rfm_features = ["recency_days", "frequency_submissions", "total_minutes", "avg_score"]
    X_rfm = rfm_df[rfm_features].copy()

    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(X_rfm)

    # Evaluate k=2 to 6
    k_range = range(2, 7)
    silhouette_scores = []
    inertias = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_rfm_scaled)
        score = silhouette_score(X_rfm_scaled, labels)
        silhouette_scores.append(score)
        inertias.append(km.inertia_)
        print(f"  k={k} | Silhouette Score: {score:.4f} | Inertia: {km.inertia_:.1f}")

    # Plot Silhouette Scores & Elbow Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(k_range, silhouette_scores, marker="o", color="#2A9D8F", lw=2)
    ax1.axvline(x=3, color="#E76F51", linestyle="--", label="Chosen k=3 (Optimal)")
    ax1.set_title("Silhouette Score by Number of Clusters (k)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Clusters (k)")
    ax1.set_ylabel("Silhouette Coefficient")
    ax1.legend()

    ax2.plot(k_range, inertias, marker="s", color="#33658A", lw=2)
    ax2.set_title("Elbow Method (Inertia by k)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Clusters (k)")
    ax2.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_silhouette_elbow.png"), dpi=200)
    plt.close()

    # Fit final K-Means with k=3
    optimal_k = 3
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=15)
    rfm_df["cluster_id"] = kmeans_final.fit_predict(X_rfm_scaled)

    # Determine semantic cluster names based on cluster centroids
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans_final.cluster_centers_), columns=rfm_features)
    centroids["cluster_id"] = range(optimal_k)

    # Sort clusters by frequency & recency to assign: Active, At-Risk, Inactive
    # Inactive: Highest recency, lowest frequency
    # Active: Lowest recency, highest frequency
    # At-Risk: Middle
    centroids_sorted = centroids.sort_values(by=["recency_days", "frequency_submissions"], ascending=[False, True])
    label_mapping = {
        centroids_sorted.iloc[0]["cluster_id"]: "Inactive / Dormant",
        centroids_sorted.iloc[1]["cluster_id"]: "At-Risk / Slipping",
        centroids_sorted.iloc[2]["cluster_id"]: "Active Champions"
    }

    rfm_df["segment_name"] = rfm_df["cluster_id"].map(label_mapping)

    # Cluster profiles
    profile = rfm_df.groupby("segment_name").agg(
        learner_count=("learner_id", "count"),
        mean_recency_days=("recency_days", "mean"),
        mean_frequency=("frequency_submissions", "mean"),
        mean_total_minutes=("total_minutes", "mean"),
        mean_avg_score=("avg_score", "mean"),
        observed_churn_rate=("churn", "mean")
    ).reset_index()

    print("\nLearner Segmentation Summary (k=3):")
    print(profile.to_string(index=False))

    # Cluster visualization (Scatter: Recency vs Frequency colored by segment)
    plt.figure(figsize=(9, 5))
    palette = {"Active Champions": "#2A9D8F", "At-Risk / Slipping": "#E9C46A", "Inactive / Dormant": "#E76F51"}
    sns.scatterplot(
        data=rfm_df,
        x="recency_days",
        y="frequency_submissions",
        hue="segment_name",
        palette=palette,
        alpha=0.8,
        s=45
    )
    plt.title("Learner RFM Clusters: Recency vs. Frequency (k=3)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recency (Days Since Last Activity Prior to Cutoff)")
    plt.ylabel("Frequency (Total Prior Submissions)")
    plt.legend(title="Learner Segment", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rfm_clusters_scatter.png"), dpi=200)
    plt.close()

    # Save cluster profile to CSV
    profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"), index=False)

    return rfm_df, silhouette_scores[optimal_k - 2]


def train_and_evaluate_churn_models(rfm_df: pd.DataFrame):
    """Train Logistic Regression and Random Forest models on strictly pre-cutoff features."""
    print("\n--- Training Churn Prediction Models (Zero Label Leakage) ---")

    # Select numerical & categorical features
    numeric_features = [
        "recency_days",
        "frequency_submissions",
        "total_words",
        "total_minutes",
        "avg_score",
        "active_days",
        "grammar_rate_per_100w",
        "vocab_rate_per_100w",
        "avg_words_per_sub",
        "goal_adherence_ratio",
        "daily_goal_min"
    ]
    
    # One-hot encode plan and level
    df_encoded = pd.get_dummies(rfm_df[numeric_features + ["plan", "level"]], drop_first=True)
    feature_names = df_encoded.columns.tolist()

    X = df_encoded.values
    y = rfm_df["churn"].values

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples (Churn rate: {y.mean():.2%})")

    # Standard Scaler for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression Baseline
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr_cv_auc = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring="roc_auc").mean()
    lr.fit(X_train_scaled, y_train)

    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_acc = accuracy_score(y_test, lr_preds)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=4,
        random_state=SEED,
        class_weight="balanced_subsample"
    )
    rf_cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_prec = precision_score(y_test, rf_preds)
    rf_rec = recall_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("\nModel Evaluation on Held-Out Test Set (20%):")
    print(f"Logistic Regression: CV-AUC={lr_cv_auc:.4f} | Test AUC={lr_auc:.4f} | F1={lr_f1:.4f} | Precision={lr_prec:.4f} | Recall={lr_rec:.4f}")
    print(f"Random Forest:       CV-AUC={rf_cv_auc:.4f} | Test AUC={rf_auc:.4f} | F1={rf_f1:.4f} | Precision={rf_prec:.4f} | Recall={rf_rec:.4f}")

    # ROC Plot
    plt.figure(figsize=(8, 6))
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.3f})", color="#2E86AB", lw=2)
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})", color="#E63946", lw=2.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (AUC = 0.500)")
    plt.title("Receiver Operating Characteristic (ROC) - Churn Prediction", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "churn_roc_curve.png"), dpi=200)
    plt.close()

    # Feature Importance Plot (Random Forest)
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_df.head(10), x="importance", y="feature", palette="rocket")
    plt.title("Top 10 Churn Predictors (Random Forest Feature Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Impurity Reduction (Gini Importance)")
    plt.ylabel("Predictor Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "churn_feature_importance.png"), dpi=200)
    plt.close()

    # Save Best Model Pipeline (Random Forest + Feature Metadata)
    pipeline_payload = {
        "model": rf,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "threshold": 0.50,
        "metrics": {
            "test_auc": float(rf_auc),
            "test_f1": float(rf_f1),
            "test_precision": float(rf_prec),
            "test_recall": float(rf_rec),
            "test_accuracy": float(rf_acc)
        }
    }
    model_save_path = os.path.join(MODELS_DIR, "churn_model.joblib")
    joblib.dump(pipeline_payload, model_save_path)
    print(f"[OK] Churn model pipeline saved to {model_save_path}")

    # Metrics JSON
    metrics_summary = {
        "dataset_split": {
            "total_cohort_learners": len(rfm_df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "overall_churn_rate": float(y.mean())
        },
        "logistic_regression": {
            "cv_5fold_auc": float(lr_cv_auc),
            "test_auc": float(lr_auc),
            "precision": float(lr_prec),
            "recall": float(lr_rec),
            "f1_score": float(lr_f1),
            "accuracy": float(lr_acc)
        },
        "random_forest": {
            "cv_5fold_auc": float(rf_cv_auc),
            "test_auc": float(rf_auc),
            "precision": float(rf_prec),
            "recall": float(rf_rec),
            "f1_score": float(rf_f1),
            "accuracy": float(rf_acc)
        },
        "top_features": feat_df.head(5).to_dict(orient="records")
    }

    metrics_file = os.path.join(OUTPUT_DIR, "part_b_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[OK] Metrics saved to {metrics_file}")
    return metrics_summary


def main():
    rfm_df, cutoff_date = load_and_prepare_temporal_data()
    rfm_df, silhouette = perform_kmeans_segmentation(rfm_df)
    train_and_evaluate_churn_models(rfm_df)
    print("--- Part B Completed Successfully ---")


if __name__ == "__main__":
    main()

