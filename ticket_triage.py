"""
StorySprout Capstone - Part C: Customer Support Ticket Triage & Routing Engine
Builds TF-IDF text classification pipelines for 5 ticket categories.
Compares:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Calibrated Linear Support Vector Classifier (LinearSVC)
Implements:
- Automated department routing
- Rule-based priority scoring (Urgent / High / Medium / Low)
- Low-confidence fallback to Human Review Specialist Queue (threshold < 0.60)
- Separate holdout test on 25 hand-written unseen tickets with challenging phrasing
Saves model pipeline to models/triage_model.joblib and outputs to outputs/part_c/.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_mpl")
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "part_c")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

SEED = 42

CATEGORIES = [
    "Account & Billing",
    "Bug Report",
    "Course Content",
    "Pedagogy & Feedback",
    "Technical Support"
]

ROUTING_MAP = {
    "Account & Billing": {
        "department": "Billing & Payments Operations",
        "queue": "queue_billing_ops",
        "sla_hours": 12
    },
    "Bug Report": {
        "department": "Frontend & Mobile Engineering",
        "queue": "queue_eng_bugs",
        "sla_hours": 8
    },
    "Course Content": {
        "department": "Curriculum & Localization Team",
        "queue": "queue_curriculum_intl",
        "sla_hours": 24
    },
    "Pedagogy & Feedback": {
        "department": "Language Coaching & AI Pedagogy",
        "queue": "queue_pedagogy_feedback",
        "sla_hours": 24
    },
    "Technical Support": {
        "department": "IT Support & Cloud Infrastructure",
        "queue": "queue_it_support",
        "sla_hours": 12
    }
}

CONFIDENCE_THRESHOLD = 0.60

# 25 hand-written unseen tickets with colloquial phrasing, typos, and novel sentence structures
UNSEEN_HOLDOUT_TICKETS = [
    # Account & Billing (5)
    {"text": "Hey there, I just saw two separate deductions of $14.99 on my Visa statement today.", "true_category": "Account & Billing"},
    {"text": "Can someone please email my company an official VAT-compliant receipt for tax filing?", "true_category": "Account & Billing"},
    {"text": "My subscription upgraded without me confirming the year plan, I want my money back please.", "true_category": "Account & Billing"},
    {"text": "Is there any way to apply my university discount code retroactively to last week's charge?", "true_category": "Account & Billing"},
    {"text": "Payment declined because my debit card expired, where can I re-enter valid card numbers?", "true_category": "Account & Billing"},

    # Bug Report (5)
    {"text": "The app straight up crashes whenever I click the mic button on the German practice page.", "true_category": "Bug Report"},
    {"text": "There is a weird visual glitch on iPhone 15 where the submit arrow overlaps the text box.", "true_category": "Bug Report"},
    {"text": "Whenever I submit more than 2 paragraphs in French, the screen turns pitch black and freezes.", "true_category": "Bug Report"},
    {"text": "The audio pronunciation player plays zero sound on safari browser version 17.", "true_category": "Bug Report"},
    {"text": "Clicking on translation toggle causes an uncaught javascript error in the dev tools.", "true_category": "Bug Report"},

    # Course Content (5)
    {"text": "In the French lesson about train travel, the translation calls the conductor a baker.", "true_category": "Course Content"},
    {"text": "Can you folks add a module with medical Spanish vocabulary for emergency responders?", "true_category": "Course Content"},
    {"text": "The Japanese story uses obsolete archaic kanji that even native speakers in Tokyo do not use.", "true_category": "Course Content"},
    {"text": "Question number 4 has no right answer because options A and C are word-for-word identical.", "true_category": "Course Content"},
    {"text": "The German B2 workplace vocabulary list is missing standard terms for remote working contracts.", "true_category": "Course Content"},

    # Pedagogy & Feedback (5)
    {"text": "The AI bot is directly spitting out the full correct sentence instead of giving me a subtle clue.", "true_category": "Pedagogy & Feedback"},
    {"text": "I really prefer having the grammar rule explanations presented in English rather than pure target language.", "true_category": "Pedagogy & Feedback"},
    {"text": "The writing coach marked my subjunctive mood as wrong, but according to Cervantes institute it is right.", "true_category": "Pedagogy & Feedback"},
    {"text": "It would be wonderful if the system tracked my weak prepositions week-over-week in a summary.", "true_category": "Pedagogy & Feedback"},
    {"text": "The reading comprehension questions feel way too basic for someone studying for the DELE C1 exam.", "true_category": "Pedagogy & Feedback"},

    # Technical Support (5)
    {"text": "I keep getting an HTTP 403 Forbidden alert whenever I attempt to log in using my Google account.", "true_category": "Technical Support"},
    {"text": "None of my saved writing exercises are syncing from my iPad over to my work Mac.", "true_category": "Technical Support"},
    {"text": "I requested a password reset link three times already but nothing ever arrives in spam or inbox.", "true_category": "Technical Support"},
    {"text": "Can StorySprout be configured to work behind our strict corporate VPN proxy firewall?", "true_category": "Technical Support"},
    {"text": "The streaming websocket keeps closing abruptly after 30 seconds of inactivity.", "true_category": "Technical Support"}
]


def assign_priority(text: str) -> dict:
    """Rule-based priority scoring based on urgency cues, sentiment, and business impact."""
    text_lower = text.lower()
    
    urgent_patterns = [
        r"\b(urgent|immediately|asap|emergency)\b",
        r"\b(charged twice|double charge|unauthorized|fraud)\b",
        r"\b(crashes|crash|freezes|cannot log in|blocked|locked out)\b",
        r"\b(exam tomorrow|test in 2 hours|deadline)\b"
    ]
    high_patterns = [
        r"\b(refund|cancel|billing issue|money back)\b",
        r"\b(broken|error 403|blank screen|failing)\b",
        r"\b(not syncing|lost progress|missing data)\b"
    ]
    
    for pat in urgent_patterns:
        if re.search(pat, text_lower):
            return {"priority": "Urgent", "priority_score": 4, "escalation_flag": True}

    for pat in high_patterns:
        if re.search(pat, text_lower):
            return {"priority": "High", "priority_score": 3, "escalation_flag": False}

    if len(text.split()) > 40:
        return {"priority": "Medium", "priority_score": 2, "escalation_flag": False}

    return {"priority": "Normal", "priority_score": 1, "escalation_flag": False}


def build_and_compare_models():
    """Train and evaluate 3 text classifiers using 5-fold cross-validation."""
    print("--- Starting Part C: Support Ticket Triage Model Training ---")

    tickets_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tickets_clean.csv"))
    print(f"Total cleaned tickets available: {len(tickets_df):,}")

    X = np.array(tickets_df["ticket_text"].astype(str).tolist(), dtype=object)
    y = np.array(tickets_df["category"].astype(str).tolist(), dtype=object)

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"Training set: {len(X_train):,} | Held-out test set: {len(X_test):,}")

    # Define model candidates with identical TF-IDF feature extraction
    tfidf_config = dict(
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        stop_words="english"
    )

    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED))
        ]),
        "Calibrated LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3))
        ])
    }

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}

    print("\n--- 5-Fold Cross-Validation Performance Comparison ---")
    for name, pipeline in models.items():
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "cv_precision_mean": float(scores["test_precision_macro"].mean()),
            "cv_recall_mean": float(scores["test_recall_macro"].mean())
        }
        print(f"[{name}] Acc: {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f}) | F1 Macro: {scores['test_f1_macro'].mean():.4f}")

    # Evaluate on held-out test split (20%)
    test_evaluations = {}
    fitted_pipelines = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        
        test_evaluations[name] = {
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1),
            "test_precision_macro": float(prec),
            "test_recall_macro": float(rec)
        }

    # Select best model based on F1-macro score
    best_model_name = max(test_evaluations, key=lambda k: test_evaluations[k]["test_f1_macro"])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n[WINNER] Selected Best Model: {best_model_name} (Test F1: {test_evaluations[best_model_name]['test_f1_macro']:.4f})")

    # Generate Confusion Matrix on Test Set
    y_test_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=CATEGORIES)
    
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f"Test Confusion Matrix - {best_model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("Ground Truth Category", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "triage_confusion_matrix.png"), dpi=200)
    plt.close()

    # Model Comparison Chart
    plt.figure(figsize=(9, 5))
    comp_df = pd.DataFrame([
        {"Model": m, "Metric": "5-Fold CV Accuracy", "Score": cv_results[m]["cv_accuracy_mean"]}
        for m in models
    ] + [
        {"Model": m, "Metric": "Test Set F1 Macro", "Score": test_evaluations[m]["test_f1_macro"]}
        for m in models
    ])
    sns.barplot(data=comp_df, x="Model", y="Score", hue="Metric", palette="crest")
    plt.ylim(0.70, 1.02)
    plt.title("Support Ticket Classifier Model Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_benchmark.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # Evaluate on 25 Unseen Hand-Written Tickets (Hard generalization test)
    # -------------------------------------------------------------
    print("\n--- Evaluating on 25 Unseen Hand-Written Holdout Tickets ---")
    holdout_texts = [t["text"] for t in UNSEEN_HOLDOUT_TICKETS]
    holdout_true = [t["true_category"] for t in UNSEEN_HOLDOUT_TICKETS]

    # Predict with probabilities
    probs = best_pipeline.predict_proba(holdout_texts)
    preds = best_pipeline.classes_[np.argmax(probs, axis=1)]
    confidences = np.max(probs, axis=1)

    holdout_correct = sum(p == t for p, t in zip(preds, holdout_true))
    holdout_accuracy = holdout_correct / len(UNSEEN_HOLDOUT_TICKETS)
    holdout_f1 = f1_score(holdout_true, preds, average="macro")

    holdout_results_table = []
    for i, t in enumerate(UNSEEN_HOLDOUT_TICKETS):
        pred_cat = preds[i]
        conf = confidences[i]
        is_correct = (pred_cat == t["true_category"])
        
        # Priority and Routing
        prio_info = assign_priority(t["text"])
        routing = ROUTING_MAP.get(pred_cat, ROUTING_MAP["Technical Support"])
        
        # Check low-confidence fallback
        requires_human = (conf < CONFIDENCE_THRESHOLD)
        assigned_queue = "queue_human_specialist_review" if requires_human else routing["queue"]

        holdout_results_table.append({
            "ticket_text": t["text"],
            "true_category": t["true_category"],
            "predicted_category": pred_cat,
            "confidence": float(round(conf, 4)),
            "is_correct": bool(is_correct),
            "assigned_priority": prio_info["priority"],
            "requires_human_review": bool(requires_human),
            "final_routing_queue": assigned_queue
        })

    print(f"Unseen Holdout Accuracy: {holdout_accuracy:.2%} ({holdout_correct}/{len(UNSEEN_HOLDOUT_TICKETS)}) | F1-Macro: {holdout_f1:.4f}")

    # Save Best Model Pipeline
    model_payload = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "classes": list(best_pipeline.classes_),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "routing_map": ROUTING_MAP
    }
    model_path = os.path.join(MODELS_DIR, "triage_model.joblib")
    joblib.dump(model_payload, model_path)
    print(f"[OK] Best triage pipeline saved to {model_path}")

    # Save Metrics JSON
    metrics_export = {
        "cross_validation": cv_results,
        "test_set_evaluation": test_evaluations,
        "selected_best_model": best_model_name,
        "unseen_holdout_evaluation": {
            "total_tickets": len(UNSEEN_HOLDOUT_TICKETS),
            "accuracy": float(holdout_accuracy),
            "f1_macro": float(holdout_f1),
            "correct_predictions": int(holdout_correct),
            "low_confidence_flagged": sum(1 for r in holdout_results_table if r["requires_human_review"])
        },
        "holdout_detailed_results": holdout_results_table
    }

    metrics_file = os.path.join(OUTPUT_DIR, "part_c_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"[OK] Part C metrics saved to {metrics_file}")

    # Save Markdown Report
    generate_triage_report(metrics_export)
    print("--- Part C Completed Successfully ---")


def generate_triage_report(metrics: dict):
    """Generate structured markdown summary for Part C."""
    report = f"""# Part C: Support Ticket Triage, Routing & Generalization Report

## 1. Model Architecture & Cross-Validation Comparison
| Model | 5-Fold CV Accuracy | 5-Fold CV F1-Macro | Test Set Accuracy | Test Set F1-Macro |
| :--- | :---: | :---: | :---: | :---: |
"""
    for m in metrics["cross_validation"]:
        cv_acc = metrics["cross_validation"][m]["cv_accuracy_mean"]
        cv_f1 = metrics["cross_validation"][m]["cv_f1_macro_mean"]
        test_acc = metrics["test_set_evaluation"][m]["test_accuracy"]
        test_f1 = metrics["test_set_evaluation"][m]["test_f1_macro"]
        report += f"| **{m}** | {cv_acc:.2%} | {cv_f1:.4f} | {test_acc:.2%} | {test_f1:.4f} |\n"

    report += f"""
**Selected Model:** `{metrics['selected_best_model']}`

## 2. Hard Holdout Test (25 Unseen Hand-Written Tickets)
- **Generalization Accuracy:** {metrics['unseen_holdout_evaluation']['accuracy']:.2%} ({metrics['unseen_holdout_evaluation']['correct_predictions']}/{metrics['unseen_holdout_evaluation']['total_tickets']})
- **Holdout F1-Macro:** {metrics['unseen_holdout_evaluation']['f1_macro']:.4f}
- **Low-Confidence Fallback Triggered:** {metrics['unseen_holdout_evaluation']['low_confidence_flagged']} tickets routed to `Human Review Specialist Queue` due to confidence < {CONFIDENCE_THRESHOLD:.2f}.

## 3. SLA & Department Routing Specifications
- **Account & Billing** -> `Billing & Payments Operations` (SLA: 12h)
- **Bug Report** -> `Frontend & Mobile Engineering` (SLA: 8h)
- **Course Content** -> `Curriculum & Localization Team` (SLA: 24h)
- **Pedagogy & Feedback** -> `Language Coaching & AI Pedagogy` (SLA: 24h)
- **Technical Support** -> `IT Support & Cloud Infrastructure` (SLA: 12h)
"""
    with open(os.path.join(OUTPUT_DIR, "triage_report.md"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    build_and_compare_models()

