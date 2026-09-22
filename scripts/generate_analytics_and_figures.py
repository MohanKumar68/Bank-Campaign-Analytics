"""
Bank Campaign Analytics and Term Deposit Response Prediction
Student Name: Mohan Kumar M
Internship: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
Execution Script for Analytics, Feature Engineering, Models, Figures & Metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

# Ensure output directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/model_results', exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------
print("--> Loading dataset from data/bank.csv...")
df = pd.read_csv('data/bank.csv', sep=';')
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------------
# 2. DATA QUALITY AUDIT
# -------------------------------------------------------------
null_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()
print(f"Missing Values across all columns: {null_counts.sum()}")
print(f"Duplicate records: {duplicate_count}")

# -------------------------------------------------------------
# 3. BUSINESS KPIS CALCULATION
# -------------------------------------------------------------
print("--> Computing Business KPIs...")
total_customers = len(df)
subscribed_customers = int((df['y'] == 'yes').sum())
subscription_rate = (subscribed_customers / total_customers) * 100
avg_balance = float(df['balance'].mean())
median_balance = float(df['balance'].median())
avg_contacts = float(df['campaign'].mean())
avg_duration = float(df['duration'].mean())
previously_contacted_pct = float((df['pdays'] != -1).mean() * 100)

prev_contacted_df = df[df['pdays'] != -1]
prev_success_rate = float((prev_contacted_df['poutcome'] == 'success').mean() * 100) if len(prev_contacted_df) > 0 else 0.0

kpis = {
    'Metric': [
        'Total Customers Contacted',
        'Subscribed Customers (y=yes)',
        'Overall Subscription Rate (%)',
        'Average Account Balance (€)',
        'Median Account Balance (€)',
        'Average Campaign Contacts per Customer',
        'Average Call Duration (seconds)',
        'Previously Contacted Customer Share (%)',
        'Previous Campaign Success Rate (%)'
    ],
    'Value': [
        f"{total_customers:,}",
        f"{subscribed_customers:,}",
        f"{subscription_rate:.2f}%",
        f"€{avg_balance:,.2f}",
        f"€{median_balance:,.2f}",
        f"{avg_contacts:.2f}",
        f"{avg_duration:.2f} s ({avg_duration/60:.2f} min)",
        f"{previously_contacted_pct:.2f}%",
        f"{prev_success_rate:.2f}%"
    ]
}
kpi_df = pd.DataFrame(kpis)
kpi_df.to_csv('outputs/model_results/business_kpis.csv', index=False)
print("Saved Business KPIs to outputs/model_results/business_kpis.csv")

# -------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------------------
print("--> Engineering domain features...")
# Age grouping
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 30, 40, 50, 60, 120],
    labels=['<30', '30-39', '40-49', '50-59', '60+'],
    right=False
)

# Balance grouping
def categorize_balance(b):
    if b < 0:
        return 'Negative'
    elif b <= 500:
        return 'Low (0-500)'
    elif b <= 2000:
        return 'Medium (501-2000)'
    else:
        return 'High (>2000)'

df['balance_group'] = df['balance'].apply(categorize_balance)

# Previously contacted flag
df['previously_contacted'] = np.where(df['pdays'] == -1, 'No', 'Yes')

# Contact frequency grouping
def categorize_contacts(c):
    if c == 1:
        return '1 contact'
    elif c <= 3:
        return '2-3 contacts'
    else:
        return '4+ contacts'

df['contact_frequency'] = df['campaign'].apply(categorize_contacts)

# Binary target variable
df['target'] = (df['y'] == 'yes').astype(int)

# -------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# -------------------------------------------------------------
print("--> Generating 8 EDA figures...")

# Figure 1: Target Distribution
fig, ax = plt.subplots(figsize=(7, 5))
counts = df['y'].value_counts()
colors = ['#3b82f6', '#10b981']
bars = ax.bar(['No (Did not subscribe)', 'Yes (Subscribed)'], counts.values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
ax.set_title('Figure 1: Term Deposit Subscription Target Distribution (y)', pad=15)
ax.set_ylabel('Number of Customers')
ax.set_ylim(0, 4600)
for bar in bars:
    height = bar.get_height()
    pct = (height / total_customers) * 100
    ax.annotate(f'{height:,}\n({pct:.2f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/eda_target_distribution.png', dpi=300)
plt.close()

# Figure 2: Subscription Rate by Job
fig, ax = plt.subplots(figsize=(10, 6))
job_rates = df.groupby('job')['target'].agg(['mean', 'count']).sort_values(by='mean', ascending=True)
bars = ax.barh(job_rates.index, job_rates['mean'] * 100, color='#2563eb', edgecolor='black', linewidth=0.6)
ax.axvline(subscription_rate, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Overall Baseline ({subscription_rate:.1f}%)')
ax.set_title('Figure 2: Term Deposit Subscription Rate by Job Category', pad=15)
ax.set_xlabel('Subscription Rate (%)')
ax.set_ylabel('Job Category')
ax.set_xlim(0, 30)
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.1f}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/figures/eda_job_subscription.png', dpi=300)
plt.close()

# Figure 3: Subscription Rate by Education Level
fig, ax = plt.subplots(figsize=(7, 5))
edu_rates = df.groupby('education')['target'].mean().reindex(['primary', 'secondary', 'tertiary', 'unknown']) * 100
bars = ax.bar(edu_rates.index.str.capitalize(), edu_rates.values, color='#0891b2', edgecolor='black', width=0.5)
ax.axhline(subscription_rate, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Baseline ({subscription_rate:.1f}%)')
ax.set_title('Figure 3: Subscription Rate by Education Level', pad=15)
ax.set_ylabel('Subscription Rate (%)')
ax.set_ylim(0, 20)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('outputs/figures/eda_education_subscription.png', dpi=300)
plt.close()

# Figure 4: Subscription Rate by Housing & Personal Loan Status
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
hl_rates = df.groupby('housing')['target'].mean() * 100
bars1 = ax1.bar(['No Housing Loan', 'Has Housing Loan'], [hl_rates['no'], hl_rates['yes']], color=['#10b981', '#f59e0b'], edgecolor='black', width=0.45)
ax1.set_title('By Housing Loan Status')
ax1.set_ylabel('Subscription Rate (%)')
ax1.set_ylim(0, 20)
for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

pl_rates = df.groupby('loan')['target'].mean() * 100
bars2 = ax2.bar(['No Personal Loan', 'Has Personal Loan'], [pl_rates['no'], pl_rates['yes']], color=['#10b981', '#f59e0b'], edgecolor='black', width=0.45)
ax2.set_title('By Personal Loan Status')
ax2.set_ylabel('Subscription Rate (%)')
ax2.set_ylim(0, 20)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

fig.suptitle('Figure 4: Impact of Existing Liabilities on Subscription Rate', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/eda_housing_loan.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Balance Distribution by Subscription Status
fig, ax = plt.subplots(figsize=(8, 5))
# Clip extreme balance outliers for visual clarity in distribution representation
clipped_balance = df['balance'].clip(lower=-1000, upper=10000)
plot_df = pd.DataFrame({'balance': clipped_balance, 'Subscribed': df['y'].map({'yes': 'Yes (Subscribed)', 'no': 'No (Did not subscribe)'})})
sns.boxplot(x='Subscribed', y='balance', data=plot_df, palette=['#3b82f6', '#10b981'], ax=ax, width=0.4)
ax.set_title('Figure 5: Account Balance Distribution by Subscription Outcome', pad=15)
ax.set_ylabel('Account Balance (€, clipped at [-1,000, +10,000] for display)')
ax.set_xlabel('Subscription Outcome')
plt.tight_layout()
plt.savefig('outputs/figures/eda_balance_distribution.png', dpi=300)
plt.close()

# Figure 6: Campaign Contacts vs Subscription Rate
fig, ax = plt.subplots(figsize=(8, 5))
contact_bins = [0, 1, 2, 3, 4, 6, 10, 50]
labels = ['1', '2', '3', '4', '5-6', '7-10', '>10']
df['contact_bin'] = pd.cut(df['campaign'], bins=contact_bins, labels=labels)
campaign_summary = df.groupby('contact_bin', observed=False)['target'].agg(['mean', 'count'])
bars = ax.bar(campaign_summary.index, campaign_summary['mean'] * 100, color='#8b5cf6', edgecolor='black', width=0.55)
ax.axhline(subscription_rate, color='#ef4444', linestyle='--', label=f'Baseline ({subscription_rate:.1f}%)')
ax.set_title('Figure 6: Subscription Rate vs Number of Campaign Contacts', pad=15)
ax.set_xlabel('Number of Contacts During Current Campaign')
ax.set_ylabel('Subscription Rate (%)')
ax.set_ylim(0, 20)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outputs/figures/eda_campaign_contacts.png', dpi=300)
plt.close()

# Figure 7: Previous Campaign Outcome vs Subscription Rate
fig, ax = plt.subplots(figsize=(7, 5))
poutcome_rates = df.groupby('poutcome')['target'].agg(['mean', 'count']).reindex(['unknown', 'failure', 'other', 'success'])
bars = ax.bar(poutcome_rates.index.str.capitalize(), poutcome_rates['mean'] * 100, color=['#94a3b8', '#f87171', '#fbbf24', '#34d399'], edgecolor='black', width=0.5)
ax.axhline(subscription_rate, color='#ef4444', linestyle='--', label=f'Baseline ({subscription_rate:.1f}%)')
ax.set_title('Figure 7: Subscription Rate by Previous Campaign Outcome (poutcome)', pad=15)
ax.set_xlabel('Outcome of Previous Marketing Campaign')
ax.set_ylabel('Subscription Rate (%)')
ax.set_ylim(0, 75)
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('outputs/figures/eda_poutcome_subscription.png', dpi=300)
plt.close()

# Figure 8: Duration Analysis (Exploratory / Post-Contact Observation)
fig, ax = plt.subplots(figsize=(8, 5))
duration_df = df.copy()
duration_df['duration_min'] = duration_df['duration'] / 60.0
sns.kdeplot(data=duration_df, x='duration_min', hue='y', common_norm=False, fill=True, palette={'no': '#3b82f6', 'yes': '#10b981'}, ax=ax, alpha=0.4)
ax.set_title('Figure 8: Contact Duration Distribution (Post-Contact Variable)', pad=15)
ax.set_xlabel('Call Duration (Minutes)')
ax.set_ylabel('Density')
ax.set_xlim(0, 25)
ax.text(0.55, 0.75, "CRITICAL NOTE:\nDuration is only known AFTER the call.\nExcluded from pre-contact ML to avoid leakage!",
        transform=ax.transAxes, fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor='#ef4444'))
plt.tight_layout()
plt.savefig('outputs/figures/eda_duration_analysis.png', dpi=300)
plt.close()

print("All 8 EDA figures successfully saved to outputs/figures/")

# -------------------------------------------------------------
# 6. DATA PREPARATION & LEAKAGE EXPERIMENT
# -------------------------------------------------------------
print("--> Setting up Features & Train-Test Splits...")

# Categorical and numerical column sets
pre_contact_cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age_group', 'balance_group', 'previously_contacted', 'contact_frequency']
pre_contact_num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

# Full feature sets
X_pre = df[pre_contact_cat_cols + pre_contact_num_cols]
y = df['target']

# With duration (for leakage demonstration)
leak_cat_cols = pre_contact_cat_cols
leak_num_cols = pre_contact_num_cols + ['duration']
X_leak = df[leak_cat_cols + leak_num_cols]

# Stratified 80-20 Train-Test split
X_train_pre, X_test_pre, y_train, y_test = train_test_split(
    X_pre, y, test_size=0.20, random_state=42, stratify=y
)
X_train_leak, X_test_leak, _, _ = train_test_split(
    X_leak, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {len(X_train_pre)} records ({y_train.sum()} positive)")
print(f"Testing set:  {len(X_test_pre)} records ({y_test.sum()} positive)")

# Pipelines
pre_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), pre_contact_num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), pre_contact_cat_cols)
    ]
)

leak_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), leak_num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), leak_cat_cols)
    ]
)

# -------------------------------------------------------------
# 7. DATA LEAKAGE DEMONSTRATION EXPERIMENT
# -------------------------------------------------------------
print("--> Executing Data Leakage Demonstration Experiment...")
# Model with duration (Leakage)
rf_leak = Pipeline([
    ('prep', leak_transformer),
    ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])
rf_leak.fit(X_train_leak, y_train)
y_pred_leak = rf_leak.predict(X_test_leak)
y_prob_leak = rf_leak.predict_proba(X_test_leak)[:, 1]

acc_leak = accuracy_score(y_test, y_pred_leak)
prec_leak = precision_score(y_test, y_pred_leak)
rec_leak = recall_score(y_test, y_pred_leak)
f1_leak = f1_score(y_test, y_pred_leak)
auc_leak = roc_auc_score(y_test, y_prob_leak)

# Model without duration (Clean Pre-Contact)
rf_pre = Pipeline([
    ('prep', pre_transformer),
    ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])
rf_pre.fit(X_train_pre, y_train)
y_pred_pre_rf = rf_pre.predict(X_test_pre)
y_prob_pre_rf = rf_pre.predict_proba(X_test_pre)[:, 1]

acc_pre_rf = accuracy_score(y_test, y_pred_pre_rf)
prec_pre_rf = precision_score(y_test, y_pred_pre_rf)
rec_pre_rf = recall_score(y_test, y_pred_pre_rf)
f1_pre_rf = f1_score(y_test, y_pred_pre_rf)
auc_pre_rf = roc_auc_score(y_test, y_prob_pre_rf)

leak_comp_df = pd.DataFrame({
    'Model Configuration': ['With Duration (Data Leakage)', 'Without Duration (Realistic Pre-Contact)'],
    'Accuracy': [acc_leak, acc_pre_rf],
    'Precision': [prec_leak, prec_pre_rf],
    'Recall': [rec_leak, rec_pre_rf],
    'F1-score': [f1_leak, f1_pre_rf],
    'ROC-AUC': [auc_leak, auc_pre_rf]
})
leak_comp_df.to_csv('outputs/model_results/leakage_experiment_results.csv', index=False)
print("Saved leakage experiment results to outputs/model_results/leakage_experiment_results.csv")

# Plot Leakage Comparison ROC curves
fig, ax = plt.subplots(figsize=(7, 5.5))
fpr_leak, tpr_leak, _ = roc_curve(y_test, y_prob_leak)
fpr_clean, tpr_clean, _ = roc_curve(y_test, y_prob_pre_rf)

ax.plot(fpr_leak, tpr_leak, color='#dc2626', lw=2.5, label=f'With Duration (Leaked AUC = {auc_leak:.3f})')
ax.plot(fpr_clean, tpr_clean, color='#2563eb', lw=2.5, label=f'Without Duration (Valid AUC = {auc_pre_rf:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
ax.set_title('Figure 9: Impact of Data Leakage (Call Duration) on ROC Performance', pad=15)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/figures/leakage_comparison.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 8. PRE-CONTACT MODEL 1: LOGISTIC REGRESSION
# -------------------------------------------------------------
print("--> Training Pre-contact Model 1: Logistic Regression...")
lr_pipeline = Pipeline([
    ('prep', pre_transformer),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train_pre, y_train)

y_pred_lr = lr_pipeline.predict(X_test_pre)
y_prob_lr = lr_pipeline.predict_proba(X_test_pre)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

# Plot Confusion Matrix for Logistic Regression
fig, ax = plt.subplots(figsize=(6, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['No Sub (0)', 'Subscribed (1)'],
            yticklabels=['No Sub (0)', 'Subscribed (1)'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Figure 10: Confusion Matrix - Logistic Regression (No Duration)', pad=15)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual True Label')
plt.tight_layout()
plt.savefig('outputs/figures/model_cm_logistic_regression.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 9. PRE-CONTACT MODEL 2: RANDOM FOREST
# -------------------------------------------------------------
print("--> Evaluating Pre-contact Model 2: Random Forest...")
# (Already trained rf_pre above)
# Plot Confusion Matrix for Random Forest
fig, ax = plt.subplots(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, y_pred_pre_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax,
            xticklabels=['No Sub (0)', 'Subscribed (1)'],
            yticklabels=['No Sub (0)', 'Subscribed (1)'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title('Figure 11: Confusion Matrix - Random Forest (No Duration)', pad=15)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual True Label')
plt.tight_layout()
plt.savefig('outputs/figures/model_cm_random_forest.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 10. MODEL COMPARISON (WITHOUT DURATION)
# -------------------------------------------------------------
print("--> Compiling Model Comparison Metrics...")
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression (Balanced)', 'Random Forest (Balanced)'],
    'Accuracy': [round(acc_lr, 4), round(acc_pre_rf, 4)],
    'Precision': [round(prec_lr, 4), round(prec_pre_rf, 4)],
    'Recall': [round(rec_lr, 4), round(rec_pre_rf, 4)],
    'F1-score': [round(f1_lr, 4), round(f1_pre_rf, 4)],
    'ROC-AUC': [round(auc_lr, 4), round(auc_pre_rf, 4)]
})
comparison_df.to_csv('outputs/model_results/model_comparison_metrics.csv', index=False)
print("Saved model comparison to outputs/model_results/model_comparison_metrics.csv:")
print(comparison_df.to_string(index=False))

# Combined ROC Curve
fig, ax = plt.subplots(figsize=(7, 5.5))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_pre_rf)
ax.plot(fpr_lr, tpr_lr, color='#3b82f6', lw=2.2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax.plot(fpr_rf, tpr_rf, color='#10b981', lw=2.2, label=f'Random Forest (AUC = {auc_pre_rf:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.500)')
ax.set_title('Figure 12: Pre-Contact Models ROC Curve Comparison', pad=15)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/figures/model_roc_curves.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 11. MODEL INTERPRETATION
# -------------------------------------------------------------
print("--> Extracting Feature Interpretations...")

# Get feature names from ColumnTransformer
ohe_feature_names = rf_pre.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(pre_contact_cat_cols)
all_feature_names = pre_contact_num_cols + list(ohe_feature_names)

# Random Forest Feature Importance
rf_importances = rf_pre.named_steps['model'].feature_importances_
rf_feat_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)
rf_feat_df.to_csv('outputs/model_results/rf_feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 6))
top15_rf = rf_feat_df.head(15).iloc[::-1]
bars = ax.barh(top15_rf['Feature'], top15_rf['Importance'], color='#059669', edgecolor='black', linewidth=0.6)
ax.set_title('Figure 13: Top 15 Feature Importances - Random Forest (Pre-Contact)', pad=15)
ax.set_xlabel('Relative Importance (Gini Impurity Reduction)')
plt.tight_layout()
plt.savefig('outputs/figures/model_rf_feature_importance.png', dpi=300)
plt.close()

# Logistic Regression Coefficients
lr_coefs = lr_pipeline.named_steps['model'].coef_[0]
lr_coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': lr_coefs
}).sort_values(by='Coefficient', ascending=False)
lr_coef_df.to_csv('outputs/model_results/lr_coefficients.csv', index=False)

# Top 10 positive and top 10 negative coefficients
top_pos = lr_coef_df.head(7)
top_neg = lr_coef_df.tail(7)
top_lr = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#ef4444' if c < 0 else '#10b981' for c in top_lr['Coefficient']]
ax.barh(top_lr['Feature'], top_lr['Coefficient'], color=colors, edgecolor='black', linewidth=0.6)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Figure 14: Top Influential Coefficients - Logistic Regression (Log-Odds)', pad=15)
ax.set_xlabel('Coefficient Value (Log-Odds Impact)')
plt.tight_layout()
plt.savefig('outputs/figures/model_lr_coefficients.png', dpi=300)
plt.close()

print("--> All analytical steps completed successfully!")
