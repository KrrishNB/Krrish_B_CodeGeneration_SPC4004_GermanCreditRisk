"""
German Credit Dataset - Binary Classification (IMPROVED VERSION)
Predicting creditworthiness (good risk vs. bad risk) using Logistic Regression

Key Improvements:
- One-Hot Encoding for categorical features (instead of Label Encoding)
- ColumnTransformer + Pipeline to prevent data leakage
- Stratified K-Fold Cross-Validation for robust evaluation
- Proper preprocessing within each fold

Dataset: UCI Statlog German Credit (1,000 instances, 20 features)
Prompt 2, Code 2, Iteration 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("=" * 80)
print("GERMAN CREDIT DATASET - BINARY CLASSIFICATION (IMPROVED)")
print("=" * 80)

# Define column names for the German Credit dataset
column_names = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

# Load dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

try:
    df = pd.read_csv(url, sep=r'\s+', header=None, names=column_names)
    print(f"\n✓ Dataset loaded successfully from UCI repository!")
    print(f"  Shape: {df.shape[0]} instances, {df.shape[1]} columns")
    data_source = "UCI Repository"
except Exception as e:
    print(f"\n⚠ Could not load from UCI repository: {e}")
    print("  Attempting to load from local file...")

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

        df = pd.DataFrame({
            'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
            'duration': np.random.randint(4, 73, n_samples),
            'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
            'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'], n_samples),
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

print("\nTarget variable distribution:")
print(df['class'].value_counts())
print(f"Class balance: {df['class'].value_counts(normalize=True).round(3).to_dict()}")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n" + "-" * 80)
print("DATA PREPARATION")
print("-" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Convert target to binary (1: good credit, 2: bad credit -> 1: good, 0: bad)
y = y.map({1: 1, 2: 0})  # 1 = Good Risk, 0 = Bad Risk

print(f"\n✓ Target variable converted to binary (1=Good Risk, 0=Bad Risk)")
print(f"  - Good Risk (1): {(y == 1).sum()} instances ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  - Bad Risk (0): {(y == 0).sum()} instances ({(y == 0).sum()/len(y)*100:.1f}%)")

# Identify categorical and numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Feature types identified:")
print(f"  - Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")

# ============================================================================
# 4. BUILD PREPROCESSING PIPELINE (PREVENTS DATA LEAKAGE)
# ============================================================================
print("\n" + "-" * 80)
print("PREPROCESSING PIPELINE CONSTRUCTION")
print("-" * 80)

# Define preprocessing for numerical features: StandardScaler
numerical_transformer = StandardScaler()

# Define preprocessing for categorical features: OneHotEncoder
# handle_unknown='ignore' ensures new categories in test data don't cause errors
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine preprocessing steps using ColumnTransformer
# This ensures numerical features are scaled and categorical features are one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep any other columns as-is (though we have none)
)

print("\n✓ Preprocessing pipeline created:")
print("  - Numerical features: StandardScaler (mean=0, std=1)")
print("  - Categorical features: OneHotEncoder (handle_unknown='ignore')")
print("  - ColumnTransformer ensures proper transformation separation")

# ============================================================================
# 5. CREATE FULL PIPELINE (PREPROCESSING + MODEL)
# ============================================================================
print("\n" + "-" * 80)
print("MODEL PIPELINE CONSTRUCTION")
print("-" * 80)

# Create full pipeline: Preprocessor + Logistic Regression
# This ensures preprocessing happens INSIDE cross-validation folds
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear'  # Good for small to medium datasets
    ))
])

print("\n✓ Full pipeline created:")
print("  Pipeline Steps:")
print("    1. Preprocessor (ColumnTransformer)")
print("    2. Logistic Regression Classifier")
print("       - Solver: liblinear")
print("       - Max iterations: 1000")
print("       - Random state: 42")

# ============================================================================
# 6. TRAIN-TEST SPLIT (FOR FINAL EVALUATION)
# ============================================================================
print("\n" + "-" * 80)
print("TRAIN-TEST SPLIT")
print("-" * 80)

# Split dataset (80% train, 20% test) - stratified to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✓ Dataset split completed (stratified):")
print(f"  - Training set: {X_train.shape[0]} instances ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Testing set: {X_test.shape[0]} instances ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n  Training set class distribution:")
print(f"    Good Risk (1): {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
print(f"    Bad Risk (0): {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")

# ============================================================================
# 7. STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STRATIFIED K-FOLD CROSS-VALIDATION (k=5)")
print("=" * 80)

# Define Stratified K-Fold cross-validator
# stratified ensures each fold has similar class distribution
# shuffle=True randomizes the data before splitting
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n✓ Cross-validation setup:")
print("  - Method: Stratified K-Fold")
print("  - Number of folds: 5")
print("  - Shuffle: True")
print("  - Random state: 42")
print("  - Prevents data leakage: Preprocessing happens inside each fold\n")

# Define scoring metrics for cross-validation
scoring = {
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

# Perform cross-validation
# The pipeline ensures preprocessing is fit only on training folds
cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1  # Use all available cores
)

# Display results for each fold
print("-" * 80)
print("CROSS-VALIDATION RESULTS BY FOLD")
print("-" * 80)

for fold in range(5):
    print(f"\nFold {fold + 1}:")
    print(f"  Accuracy:  {cv_results['test_accuracy'][fold]:.4f}")
    print(f"  ROC-AUC:   {cv_results['test_roc_auc'][fold]:.4f}")
    print(f"  F1-Score:  {cv_results['test_f1'][fold]:.4f}")
    print(f"  Precision: {cv_results['test_precision'][fold]:.4f}")
    print(f"  Recall:    {cv_results['test_recall'][fold]:.4f}")

# Calculate and display average metrics across all folds
print("\n" + "=" * 80)
print("AVERAGE CROSS-VALIDATION METRICS (5 FOLDS)")
print("=" * 80)

print(f"\n✓ ACCURACY:   {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"✓ ROC-AUC:    {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"✓ F1-SCORE:   {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print(f"✓ PRECISION:  {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"✓ RECALL:     {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")

# ============================================================================
# 8. TRAIN FINAL MODEL AND EVALUATE ON TEST SET
# ============================================================================
print("\n" + "=" * 80)
print("FINAL MODEL TRAINING AND TEST SET EVALUATION")
print("=" * 80)

# Train the pipeline on the full training set
print("\n✓ Training final model on full training set...")
pipeline.fit(X_train, y_train)
print("  - Model training completed!")

# Make predictions on test set
print("\n✓ Generating predictions on test set...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

print(f"  - Predictions generated for {len(y_test)} test instances")

# Display sample predictions with probabilities
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
# 9. DETAILED TEST SET EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("TEST SET EVALUATION RESULTS")
print("=" * 80)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion Matrix
print("\n✓ CONFUSION MATRIX:")
print("-" * 40)
cm = confusion_matrix(y_test, y_pred)
print(f"\n                Predicted")
print(f"              Bad    Good")
print(f"Actual  Bad   {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"        Good  {cm[1,0]:3d}    {cm[1,1]:3d}")

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives (TN):  {tn} - Correctly identified Bad Risk")
print(f"  False Positives (FP): {fp} - Bad Risk misclassified as Good (RISKY!)")
print(f"  False Negatives (FN): {fn} - Good Risk misclassified as Bad")
print(f"  True Positives (TP):  {tp} - Correctly identified Good Risk")

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

# F1-Score
f1 = f1_score(y_test, y_pred)
print(f"✓ F1-SCORE: {f1:.4f}")

# Additional Metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n✓ ADDITIONAL METRICS:")
print(f"  - Sensitivity (Recall): {sensitivity:.4f} - Ability to identify Good Risk")
print(f"  - Specificity:          {specificity:.4f} - Ability to identify Bad Risk")
print(f"  - Precision:            {precision:.4f} - Accuracy of Good Risk predictions")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "-" * 80)
print("GENERATING VISUALIZATIONS")
print("-" * 80)

plt.style.use('default')

# -----------------------------------------------
# Confusion Matrix Visualization
# -----------------------------------------------
print("\n✓ Creating Confusion Matrix heatmap...")

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# Create heatmap
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# Set labels
ax.set(xticks=[0, 1],
       yticks=[0, 1],
       xticklabels=['Bad Risk', 'Good Risk'],
       yticklabels=['Bad Risk', 'Good Risk'],
       xlabel='Predicted Label',
       ylabel='True Label',
       title='Confusion Matrix - Logistic Regression\nGerman Credit Dataset (Improved)')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, weight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_improved.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close()
print("  - Saved as 'confusion_matrix_improved.png'")

# -----------------------------------------------
# ROC Curve Visualization
# -----------------------------------------------
print("\n✓ Creating ROC Curve...")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc_value = auc(fpr, tpr)

# Create ROC plot
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# Plot ROC curve
ax.plot(fpr, tpr, color='blue', lw=2,
        label=f'ROC Curve (AUC = {roc_auc_value:.4f})')

# Plot diagonal reference line
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
        label='Random Classifier')

# Formatting
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Logistic Regression\nGerman Credit Dataset (Improved)', fontsize=14)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('roc_curve_improved.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close()
print("  - Saved as 'roc_curve_improved.png'")

print("\n✓ Visualizations saved successfully!")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Model: Logistic Regression (Baseline - IMPROVED)
Dataset: UCI German Credit (1,000 instances, 20 features)
Data Source: {data_source}

KEY IMPROVEMENTS IMPLEMENTED:
  ✓ One-Hot Encoding for categorical features (not Label Encoding)
  ✓ ColumnTransformer for proper feature preprocessing
  ✓ Pipeline ensures no data leakage
  ✓ Stratified K-Fold Cross-Validation (k=5)
  ✓ Preprocessing happens inside each CV fold

CROSS-VALIDATION RESULTS (5-Fold Average):
  • Accuracy:   {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}
  • ROC-AUC:    {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}
  • F1-Score:   {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}
  • Precision:  {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}
  • Recall:     {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}

TEST SET RESULTS:
  • Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)
  • ROC-AUC:    {roc_auc:.4f}
  • F1-Score:   {f1:.4f}
  • Precision:  {precision:.4f}
  • Recall:     {sensitivity:.4f}
  • Specificity: {specificity:.4f}
  
Test Set Size: {len(y_test)} instances
Correctly Classified: {(y_pred == y_test).sum()} instances
Misclassified: {(y_pred != y_test).sum()} instances

CRITICAL BUSINESS INSIGHT:
  - False Positives (Bad Risk approved): {fp} cases
    → These are high-risk applicants wrongly approved (costly!)
  - Model shows {'good' if specificity > 0.6 else 'poor'} ability to identify bad risks (Specificity: {specificity:.2%})
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# 12. SAVE PREDICTIONS (OPTIONAL)
# ============================================================================
# Uncomment to save predictions to CSV

results_df = pd.DataFrame({
    'Actual_Class': y_test.values,
    'Predicted_Class': y_pred,
    'Probability_Bad_Risk': y_pred_proba[:, 0],
    'Probability_Good_Risk': y_pred_proba[:, 1]
})
results_df.to_csv('german_credit_predictions_improved.csv', index=False)
print("\n✓ Predictions saved to 'german_credit_predictions_improved.csv'")
