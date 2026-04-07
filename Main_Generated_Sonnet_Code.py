"""
German Credit Dataset - Binary Classification
Predicting creditworthiness (good risk vs. bad risk) using Logistic Regression

Dataset: UCI Statlog German Credit (1,000 instances, 20 features)
Prompt 1, Code 1, Initial Generated Sonnet Code
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("=" * 80)
print("GERMAN CREDIT DATASET - BINARY CLASSIFICATION")
print("=" * 80)

# Define column names for the German Credit dataset
column_names = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

# Try loading from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

try:
    # Load data with space delimiter
    df = pd.read_csv(url, sep=r'\s+', header=None, names=column_names)
    print(f"\n✓ Dataset loaded successfully from UCI repository!")
    print(f"  Shape: {df.shape[0]} instances, {df.shape[1]} columns")
    data_source = "UCI Repository"
except Exception as e:
    print(f"\n⚠ Could not load from UCI repository: {e}")
    print("  Loading from local file or generating sample data...")

    # Try loading from local file
    try:
        df = pd.read_csv('german.data', sep=r'\s+', header=None, names=column_names)
        print(f"\n✓ Dataset loaded from local file!")
        print(f"  Shape: {df.shape[0]} instances, {df.shape[1]} columns")
        data_source = "Local File"
    except:
        # Generate synthetic data for demonstration
        print("\n✓ Generating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 1000

        # Create synthetic data mimicking German Credit structure
        df = pd.DataFrame({
            'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
            'duration': np.random.randint(4, 73, n_samples),
            'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
            'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
                                        n_samples),
            'credit_amount': np.random.randint(250, 18500, n_samples),
            'savings_status': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
            'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
            'installment_commitment': np.random.randint(1, 5, n_samples),
            'personal_status': np.random.choice(['A91', 'A92', 'A93', 'A94', 'A95'], n_samples),
            'other_parties': np.random.choice(['A101', 'A102', 'A103'], n_samples),
            'residence_since': np.random.randint(1, 5, n_samples),
            'property_magnitude': np.random.choice(['A121', 'A122', 'A123', 'A124'], n_samples),
            'age': np.random.randint(19, 76, n_samples),
            'other_payment_plans': np.random.choice(['A141', 'A142', 'A143'], n_samples),
            'housing': np.random.choice(['A151', 'A152', 'A153'], n_samples),
            'existing_credits': np.random.randint(1, 5, n_samples),
            'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples),
            'num_dependents': np.random.randint(1, 3, n_samples),
            'own_telephone': np.random.choice(['A191', 'A192'], n_samples),
            'foreign_worker': np.random.choice(['A201', 'A202'], n_samples),
            'class': np.random.choice([1, 2], n_samples, p=[0.7, 0.3])
        })
        print(f"  Shape: {df.shape[0]} instances, {df.shape[1]} columns")
        data_source = "Synthetic Data (Demo)"

# ============================================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================================
print("\n" + "-" * 80)
print("INITIAL DATA EXPLORATION")
print("-" * 80)
print(f"Data Source: {data_source}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nTarget variable distribution:")
print(df['class'].value_counts())
print(f"\nClass balance: {df['class'].value_counts(normalize=True).round(3).to_dict()}")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "-" * 80)
print("DATA PREPROCESSING")
print("-" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Convert target to binary (1: good credit, 2: bad credit -> 0: bad, 1: good)
y = y.map({1: 1, 2: 0})  # 1 = good risk, 0 = bad risk

print(f"\n✓ Target variable converted to binary (1=Good Risk, 0=Bad Risk)")

# Identify categorical and numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Feature types identified:")
print(f"  - Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")

# Handle categorical variables using Label Encoding
print(f"\n✓ Encoding categorical variables...")

label_encoders = {}
X_encoded = X.copy()

for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"  - Encoded {len(categorical_features)} categorical features using Label Encoding")

# Check for missing values
missing_values = X_encoded.isnull().sum().sum()
print(f"\n✓ Missing values check: {missing_values} missing values found")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "-" * 80)
print("TRAIN-TEST SPLIT")
print("-" * 80)

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution in both sets
)

print(f"\n✓ Dataset split completed:")
print(f"  - Training set: {X_train.shape[0]} instances ({X_train.shape[0] / len(X) * 100:.1f}%)")
print(f"  - Testing set: {X_test.shape[0]} instances ({X_test.shape[0] / len(X) * 100:.1f}%)")
print(f"\n  Training set class distribution:")
print(f"    Good Risk (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
print(f"    Bad Risk (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")

# ============================================================================
# 5. FEATURE SCALING
# ============================================================================
print("\n" + "-" * 80)
print("FEATURE SCALING")
print("-" * 80)

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features standardized using StandardScaler")
print(f"  - Mean ≈ 0, Standard Deviation ≈ 1")

# ============================================================================
# 6. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================
print("\n" + "-" * 80)
print("MODEL TRAINING - LOGISTIC REGRESSION")
print("-" * 80)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000,  # Ensure convergence
    solver='lbfgs'
)

print("\n✓ Training Logistic Regression classifier...")
log_reg.fit(X_train_scaled, y_train)
print("  - Model training completed successfully!")

# ============================================================================
# 7. MODEL PREDICTIONS
# ============================================================================
print("\n" + "-" * 80)
print("MODEL PREDICTIONS")
print("-" * 80)

# Make predictions on test set
y_pred = log_reg.predict(X_test_scaled)

# Get predicted probabilities for both classes
y_pred_proba = log_reg.predict_proba(X_test_scaled)

print(f"\n✓ Predictions generated for {len(y_test)} test instances")
print(f"\n✓ Predicted probabilities shape: {y_pred_proba.shape}")
print(f"  - Column 0: Probability of Bad Risk (Class 0)")
print(f"  - Column 1: Probability of Good Risk (Class 1)")

# Display first 10 predictions with probabilities
print("\nSample Predictions (First 10 instances):")
print("-" * 70)
prediction_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Prob_Bad_Risk': y_pred_proba[:10, 0].round(4),
    'Prob_Good_Risk': y_pred_proba[:10, 1].round(4)
})
print(prediction_df.to_string(index=False))

# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Confusion Matrix
print("\n✓ CONFUSION MATRIX:")
print("-" * 40)
cm = confusion_matrix(y_test, y_pred)
print(f"\n                Predicted")
print(f"              Bad    Good")
print(f"Actual  Bad   {cm[0, 0]:3d}    {cm[0, 1]:3d}")
print(f"        Good  {cm[1, 0]:3d}    {cm[1, 1]:3d}")

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

# Classification Report
print("\n✓ CLASSIFICATION REPORT:")
print("-" * 40)
print(classification_report(
    y_test,
    y_pred,
    target_names=['Bad Risk (0)', 'Good Risk (1)'],
    digits=4
))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"✓ ROC-AUC SCORE: {roc_auc:.4f}")

# Additional Metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\n✓ ADDITIONAL METRICS:")
print(f"  - Sensitivity (Recall): {sensitivity:.4f}")
print(f"  - Specificity:          {specificity:.4f}")
print(f"  - Precision:            {precision:.4f}")
print(f"  - F1-Score:             {f1:.4f}")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Model: Logistic Regression (Baseline)
Dataset: UCI German Credit (1,000 instances, 20 features)
Data Source: {data_source}

Key Results:
  • Accuracy:        {accuracy * 100:.2f}%
  • ROC-AUC:         {roc_auc:.4f}
  • Precision:       {precision:.4f}
  • Recall:          {sensitivity:.4f}
  • F1-Score:        {f1:.4f}

Test Set Size: {len(y_test)} instances
Correctly Classified: {(y_pred == y_test).sum()} instances
Misclassified: {(y_pred != y_test).sum()} instances
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# 10. SAVE PREDICTIONS (OPTIONAL)
# ============================================================================
# Uncomment the following to save predictions to CSV

results_df = pd.DataFrame({
    'Actual_Class': y_test.values,
    'Predicted_Class': y_pred,
    'Probability_Bad_Risk': y_pred_proba[:, 0],
    'Probability_Good_Risk': y_pred_proba[:, 1]
})
results_df.to_csv('german_credit_predictions.csv', index=False)
print("\n✓ Predictions saved to 'german_credit_predictions.csv'")
