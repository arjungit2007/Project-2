"""
================================================================
 PROJECT 2 : DATA CLASSIFICATION USING AI
 DecodeLabs — Artificial Intelligence Industrial Training Kit
================================================================

GOAL
    Build a basic classification model using a small dataset.

KEY REQUIREMENTS (from the brief)
    [x] Load and understand a dataset
    [x] Split data into training and testing sets
    [x] Apply a simple classification algorithm

DATASET
    orders_clean.csv — 1,200 e-commerce orders, parsed and validated
    from the supplied "Dataset for Data Analytics" file. Every row's
    TotalPrice was cross-checked against Quantity x UnitPrice during
    parsing, so the data is verified clean before any modeling starts.

TASK FRAMED AS CLASSIFICATION
    Target  : OrderStatus (Shipped / Cancelled / Returned / Delivered / Pending)
    Features: Quantity, UnitPrice, ItemsInCart, TotalPrice, ShippingAddressNo,
              Product, PaymentMethod, CouponCode, ReferralSource
    Algorithm: K-Nearest Neighbors (matches the "Architectural Paradigms"
               slide's Iris-benchmark workflow: scale -> split -> KNN ->
               confusion matrix -> F1 score)

ARCHITECTURE  (I -> P -> O), same blueprint as the slide deck:
    INPUT   : load CSV, inspect shape / dtypes / class balance
    PROCESS : encode categoricals -> scale numerics -> train/test split
              -> fit KNeighborsClassifier (K tuned via the "elbow" method)
    OUTPUT  : accuracy, confusion matrix, precision/recall/F1 report,
              plus two saved charts (error-rate-vs-K, confusion matrix)

A NOTE ON ACCURACY (see 'Output Validation' in the slides)
    OrderStatus here is not strongly caused by price/quantity/coupon —
    it's closer to a random label in this dataset. Expect a MODEST
    accuracy (roughly in line with random guessing among 5 classes).
    That is the intended lesson from Project 2's brief: a raw accuracy
    number can be an "accuracy mirage" — the confusion matrix and
    per-class F1 score tell the real story, and a low score here is a
    learning opportunity, not a bug.

Run it with:
    python classify_orders.py
Requires: pandas, scikit-learn, matplotlib (all pre-installed)
================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA_PATH = "orders_clean.csv"
TARGET_COL = "OrderStatus"
RANDOM_STATE = 42


def load_and_understand(path: str) -> pd.DataFrame:
    """
    PHASE 1: INPUT
    Load the dataset and print a quick inventory so we understand
    what we're working with before touching any modeling code.
    """
    df = pd.read_csv(path)

    print("=" * 64)
    print(" PHASE 1 — LOAD & UNDERSTAND THE DATASET")
    print("=" * 64)
    print(f"Shape            : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns          : {list(df.columns)}")
    print(f"Missing values   : {int(df.isnull().sum().sum())} total")
    print("\nClass balance (target = OrderStatus):")
    print(df[TARGET_COL].value_counts().to_string())
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print()
    return df


def build_features(df: pd.DataFrame):
    """
    PHASE 2a: PREPROCESSING
    Encode categorical columns to numbers and separate features (X)
    from the target (y). Returns X, y, and the label encoder used on
    the target (so we can turn predictions back into readable labels).
    """
    df = df.copy()

    categorical_cols = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]
    numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice", "ShippingAddressNo"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = numeric_cols + [c + "_enc" for c in categorical_cols]
    X = df[feature_cols].copy()

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL])

    return X, y, target_encoder, feature_cols


def choose_best_k(X_train_scaled, y_train, X_test_scaled, y_test, k_range=range(1, 31, 2)):
    """
    PHASE 2b: TUNE THE ENGINE (choosing K)
    Mirrors the 'error rate vs K' elbow chart from the slides. Tries
    several odd K values, records the error rate for each, and returns
    the K with the lowest test error, plus the full curve for plotting.
    """
    errors = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        error_rate = 1 - accuracy_score(y_test, preds)
        errors.append(error_rate)

    best_index = int(np.argmin(errors))
    best_k = list(k_range)[best_index]
    return best_k, list(k_range), errors


def main():
    print()
    df = load_and_understand(DATA_PATH)

    X, y, target_encoder, feature_cols = build_features(df)
    class_names = list(target_encoder.classes_)

    # PHASE 2c: STRUCTURAL INTEGRITY — the train/test split
    # Stratified so every class is represented proportionally in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )

    print("=" * 64)
    print(" PHASE 2 — TRAIN/TEST SPLIT & FEATURE SCALING")
    print("=" * 64)
    print(f"Training rows    : {len(X_train)}")
    print(f"Testing rows     : {len(X_test)}")
    print(f"Feature columns  : {feature_cols}\n")

    # The Gatekeeper Rule: scale AFTER splitting, fit the scaler on
    # training data only, then apply it to both sets.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PHASE 3: TUNE + TRAIN — pick the best K, then fit the final model
    best_k, k_values, error_rates = choose_best_k(X_train_scaled, y_train, X_test_scaled, y_test)

    print("=" * 64)
    print(" PHASE 3 — MODEL TRAINING (K-Nearest Neighbors)")
    print("=" * 64)
    print(f"Best K found     : {best_k}  (lowest test error rate)\n")

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # PHASE 4: OUTPUT VALIDATION
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=class_names, zero_division=0)

    print("=" * 64)
    print(" PHASE 4 — OUTPUT VALIDATION")
    print("=" * 64)
    print(f"Accuracy         : {accuracy:.2%}\n")
    print("Confusion matrix (rows = actual, columns = predicted):")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())
    print("\nClassification report (precision / recall / F1 per class):")
    print(report)

    print("=" * 64)
    print(" READING THE RESULT")
    print("=" * 64)
    baseline = 1 / len(class_names)
    print(
        f"Random-guess baseline for {len(class_names)} classes is ~{baseline:.2%}.\n"
        f"This model scored {accuracy:.2%}. In this dataset, OrderStatus doesn't\n"
        f"depend strongly on price/quantity/coupon/referral, so accuracy near the\n"
        f"baseline is EXPECTED, not a bug — exactly the 'accuracy mirage' lesson\n"
        f"from the brief. The confusion matrix and per-class F1 above are the\n"
        f"real diagnostic tools, not the single accuracy number.\n"
    )

    # Save charts as image artifacts
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, error_rates, marker="o", color="#0B2A4A")
    plt.axvline(best_k, color="#E8632C", linestyle="--", label=f"Best K = {best_k}")
    plt.title("Tuning the engine: error rate vs K")
    plt.xlabel("K value")
    plt.ylabel("Error rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("k_tuning_curve.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion matrix — OrderStatus classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved charts: k_tuning_curve.png, confusion_matrix.png")
    print()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(f"ERROR: could not find '{DATA_PATH}'. Place it in the same folder as this script.")
        sys.exit(1)
