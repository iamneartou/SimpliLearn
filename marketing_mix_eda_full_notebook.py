# %%
"""
Full Jupyter-style Python notebook (script with cell markers) for
"Marketing Mix" EDA, hypothesis testing and visualizations.

Drop this file into JupyterLab or run in any Python environment.
Assumes the dataset is at: /mnt/data/marketing_data.csv

Sections included:
- Imports and config
- Load data & initial inspection
- Cleaning & missing-value imputation
- Feature engineering (age, total_children, total_spend, total_purchases)
- Outlier detection & treatment
- Encoding categorical variables
- Correlation heatmap
- Hypothesis tests asked in the prompt
- Visualizations for product performance, campaign acceptance, country analysis,
  children vs spend, education vs complaints
- Conclusions & recommended next steps
"""

# %%
# Imports and configuration
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Make plots look nicer in notebook
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,6)

# File path (adjust if needed)
DATA_PATH = '/mnt/data/marketing_data.csv'

# %%
# Load data
print('Loading dataset from', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Initial shape:', df.shape)

# Quick peek
print('\nColumns:')
print(df.columns.tolist())

# %%
# Initial inspection
display(df.head())
print('\nData types:')
print(df.dtypes)
print('\nMissing values summary:')
print(df.isna().sum())

# %%
# Ensure Dt_Customer parsed correctly
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    print('Dt_Customer min, max:', df['Dt_Customer'].min(), df['Dt_Customer'].max())

# %%
# Standardize column names (lowercase, no spaces)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# %%
# Data cleaning: inspect education and marital status categories
for col in ['education', 'marital_status']:
    if col in df.columns:
        print(f"\nUnique values for {col}:")
        print(df[col].value_counts(dropna=False))

# %%
# Impute missing income using median income grouped by education + marital status
if 'income' in df.columns:
    # Clean income if it's a string with $ or commas
    if df['income'].dtype == object:
        df['income_clean'] = df['income'].astype(str).str.replace('[\$,]', '', regex=True)
        df['income_clean'] = pd.to_numeric(df['income_clean'], errors='coerce')
    else:
        df['income_clean'] = df['income'].astype(float)

    # Group median
    group_cols = [c for c in ['education', 'marital_status'] if c in df.columns]
    if group_cols:
        medians = df.groupby(group_cols)['income_clean'].median().reset_index().rename(columns={'income_clean':'median_income'})
        df = df.merge(medians, on=group_cols, how='left')
        # Fill missing incomes
        df['income_imputed'] = df['income_clean']
        mask_missing = df['income_imputed'].isna()
        df.loc[mask_missing, 'income_imputed'] = df.loc[mask_missing, 'median_income']
        # Finalize
        df['income_final'] = df['income_imputed'].fillna(df['income_clean'].median())
        print('\nFilled missing income using education + marital_status medians where available.')
    else:
        df['income_final'] = df['income_clean'].fillna(df['income_clean'].median())

    # Drop helper columns
    df.drop(columns=[c for c in ['income_clean','income_imputed','median_income'] if c in df.columns], inplace=True, errors='ignore')

# %%
# Feature engineering
# Total children: combine columns like 'kidhome' and 'teenhome' if present
children_cols = [c for c in df.columns if c in ['kidhome', 'teenhome', 'no_of_children', 'children']]
if 'kidhome' in df.columns and 'teenhome' in df.columns:
    df['total_children'] = df['kidhome'].fillna(0) + df['teenhome'].fillna(0)
elif children_cols:
    # try first available
    df['total_children'] = df[children_cols[0]].fillna(0)
else:
    df['total_children'] = 0

# Age: if birth_year exists, compute age using dataset max year or current year
if 'birth_year' in df.columns:
    try:
        # Choose reference year as max Dt_Customer year if present
        if 'dt_customer' in df.columns and pd.api.types.is_datetime64_any_dtype(df['dt_customer']):
            ref_year = df['dt_customer'].dt.year.max()
        else:
            ref_year = pd.Timestamp('today').year
        df['age'] = ref_year - df['birth_year']
    except Exception:
        df['age'] = np.nan
else:
    df['age'] = np.nan

# Total spending: sum of product spend columns (common: 'wine', 'fruits', 'gold', 'meat', 'fish', 'sweet', 'vegetables')
spend_cols = [c for c in df.columns if any(p in c for p in ['wine','fruits','gold','meat','fish','sweet','vegetable','total_spend','spend'])]
# Exclude columns that are actually counts or unrelated like 'total_purchases'
spends = [c for c in spend_cols if df[c].dtype in [np.float64, np.int64] or pd.api.types.is_numeric_dtype(df[c])]
if spends:
    df['total_spend'] = df[spends].sum(axis=1)
else:
    df['total_spend'] = np.nan

# Total purchases through channels if columns present like 'web_purchases', 'store_purchases', 'catalog_purchases' or 'num_web_purchases' etc.
purchase_cols = [c for c in df.columns if any(k in c for k in ['web', 'store', 'catalog', 'online', 'num_purchases', 'purchases'])]
# Heuristic: numeric purchase-like columns
purchase_cols = [c for c in purchase_cols if df[c].dtype in [np.int64, np.float64] and (df[c].max() < 1000)]
if purchase_cols:
    df['total_purchases'] = df[purchase_cols].sum(axis=1)
else:
    # fallback to summing specific known names
    for cand in ['web_purchases', 'store_purchases', 'catalog_purchases']:
        if cand in df.columns:
            df['total_purchases'] = df.get('total_purchases', 0) + df[cand].fillna(0)
    df['total_purchases'] = df.get('total_purchases', np.nan)

print('\nFeature engineering done. Columns added:', [c for c in ['total_children','age','total_spend','total_purchases'] if c in df.columns])

# %%
# Basic descriptive statistics
display(df[['age','income_final','total_children','total_spend','total_purchases']].describe())

# %%
# Visualize distributions & outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Boxplots for selected numeric features
for col in ['total_spend','income_final','age','total_purchases']:
    if col in df.columns:
        plt.figure(figsize=(8,3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot: {col}')
        plt.show()

# Histograms
for col in ['total_spend','income_final','age','total_purchases']:
    if col in df.columns:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram: {col}')
        plt.show()

# %%
# Outlier treatment: cap numeric variables at 1st and 99th percentiles (winsorization)
winsor_cols = [c for c in ['total_spend','income_final','total_purchases'] if c in df.columns]
for c in winsor_cols:
    lower = df[c].quantile(0.01)
    upper = df[c].quantile(0.99)
    df[c+'_winsor'] = df[c].clip(lower, upper)
    print(f'Winsorized {c}: lower={lower:.2f}, upper={upper:.2f}')

# %%
# Encoding categorical variables
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
print('\nCategorical cols found:', cat_cols)

# Ordinal encoding example for education if ordered
if 'education' in df.columns:
    # Define ordering if known, else use frequency-based order
    edu_order = ['Basic', 'High School', 'Graduation', 'Master', 'PhD']
    # Ensure values match - create mapping with fallback
    df['education_clean'] = df['education'].astype(str).str.title().replace({'Na':'Unknown','nan':'Unknown'})
    encoder = OrdinalEncoder(categories=[edu_order], dtype=float)
    try:
        df['education_ord'] = encoder.fit_transform(df[['education_clean']])
    except Exception:
        # fallback: label encode by factorized rank
        df['education_ord'] = pd.factorize(df['education_clean'])[0]

# One-hot encode remaining small-cardinality categoricals
onehot_cols = [c for c in cat_cols if c not in ['education','education_clean'] and df[c].nunique() < 10]
if onehot_cols:
    df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=True)
    print('One-hot encoded:', onehot_cols)

# %%
# Correlation heatmap for numeric features
corr_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique() > 1]
plt.figure(figsize=(12,10))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation matrix')
plt.show()

# %%
# Hypothesis tests
# 1) Older people prefer shopping in-store (compare age for store shoppers vs non-store)
# We need a column indicating store purchases; try to find one
store_col_candidates = [c for c in df.columns if 'store' in c and df[c].dtype in [np.int64, np.float64]]
print('\nStore-related numeric columns found:', store_col_candidates)

if store_col_candidates:
    store_col = store_col_candidates[0]
    # Create boolean: store shopper if store_col > 0
    df['store_shopper'] = df[store_col] > 0
    # t-test for age difference
    a = df.loc[df['store_shopper'],'age'].dropna()
    b = df.loc[~df['store_shopper'],'age'].dropna()
    if len(a)>10 and len(b)>10:
        tstat, pval = stats.ttest_ind(a, b, nan_policy='omit')
        print(f"Hypothesis 1 (age difference store vs not): t={tstat:.3f}, p={pval:.3f}")
    else:
        print('Not enough data to test hypothesis 1')
else:
    print('No store purchase column found; cannot directly test hypothesis 1')

# 2) Customers with kids prefer online (compare online purchases for customers with children vs without)
online_col_candidates = [c for c in df.columns if any(k in c for k in ['web','online']) and df[c].dtype in [np.int64, np.float64]]
print('Online-related numeric columns found:', online_col_candidates)
if online_col_candidates:
    online_col = online_col_candidates[0]
    df['has_children'] = df['total_children'] > 0
    a = df.loc[df['has_children'], online_col].dropna()
    b = df.loc[~df['has_children'], online_col].dropna()
    if len(a)>10 and len(b)>10:
        tstat, pval = stats.ttest_ind(a, b, nan_policy='omit')
        print(f"Hypothesis 2 (online purchases children vs no children): t={tstat:.3f}, p={pval:.3f}")
    else:
        print('Not enough data to test hypothesis 2')
else:
    print('No online purchase column found; cannot directly test hypothesis 2')

# 3) Other distribution channels cannibalize store sales: correlation between store and other channels
if store_col_candidates and online_col_candidates:
    store = df[store_col]
    online = df[online_col]
    corr = store.corr(online)
    print(f'Hypothesis 3 correlation store vs online: {corr:.3f}')
else:
    print('Not enough channel columns to test cannibalization hypothesis')

# 4) Does the US fare better in total purchases? Test mean total_purchases between US and others
if 'country' in df.columns and 'total_purchases' in df.columns:
    df['is_us'] = df['country'].str.strip().str.upper().eq('UNITED STATES') | df['country'].str.strip().str.upper().eq('US')
    a = df.loc[df['is_us'],'total_purchases'].dropna()
    b = df.loc[~df['is_us'],'total_purchases'].dropna()
    if len(a)>10 and len(b)>10:
        tstat, pval = stats.ttest_ind(a, b, nan_policy='omit')
        print(f"Hypothesis 4 (US vs Others total_purchases): t={tstat:.3f}, p={pval:.3f}")
    else:
        print('Not enough data to test hypothesis 4')
else:
    print('Country or total_purchases column missing; cannot test hypothesis 4')

# %%
# Visualizations asked in the prompt
# Which products perform best/worst by revenue
possible_product_cols = [c for c in df.columns if any(p in c for p in ['wine','fruits','gold','meat','fish','sweet','vegetable'])]
print('\nProduct spend columns detected:', possible_product_cols)
if possible_product_cols:
    prod_revenue = df[possible_product_cols].sum().sort_values(ascending=False)
    plt.figure()
    prod_revenue.plot(kind='bar')
    plt.title('Total revenue by product')
    plt.ylabel('Revenue')
    plt.show()

# Pattern between age and last campaign acceptance
if 'last_campaign' in df.columns or 'accepted_last_campaign' in df.columns or 'accepted_campaign' in df.columns:
    # try common names
    camp_col = None
    for name in ['accepted_last_campaign','accepted_campaign','last_campaign']:
        if name in df.columns:
            camp_col = name
            break
    if camp_col is not None:
        # ensure boolean
        df['accepted_last'] = df[camp_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes','1','true','y'] else 0 if pd.notna(x) else np.nan)
        plt.figure()
        sns.boxplot(x=df['accepted_last'].dropna(), y=df['age'])
        plt.title('Age distribution by last campaign acceptance')
        plt.xlabel('Accepted last campaign (0/1)')
        plt.show()

# Which country has greatest number of customers who accepted last campaign
if 'country' in df.columns and 'accepted_last' in df.columns:
    country_accept = df.loc[df['accepted_last']==1, 'country'].value_counts().head(10)
    plt.figure()
    country_accept.plot(kind='bar')
    plt.title('Top countries by count of accepted last campaign')
    plt.ylabel('Count')
    plt.show()

# Children at home vs total spend
if 'total_children' in df.columns and 'total_spend' in df.columns:
    plt.figure()
    sns.boxplot(x=df['total_children'].astype(int), y=df['total_spend'].fillna(0))
    plt.title('Total spend by number of children at home')
    plt.xlabel('Number of children')
    plt.show()

# Education background of customers who complained in last 2 years
complaint_col_candidates = [c for c in df.columns if 'complaint' in c or 'customer_complaint' in c or 'complained' in c]
print('Complaint columns:', complaint_col_candidates)
if complaint_col_candidates and 'education' in df.columns and 'dt_customer' in df.columns:
    comp_col = complaint_col_candidates[0]
    # filter complaints in last 2 years relative to max Dt_Customer
    if pd.api.types.is_datetime64_any_dtype(df['dt_customer']):
        ref_date = df['dt_customer'].max()
        cutoff = ref_date - pd.DateOffset(years=2)
        recent = df[(df[comp_col].notna()) & (df['dt_customer'] >= cutoff)]
        if not recent.empty:
            display(recent['education'].value_counts())
        else:
            print('No recent complaints in last 2 years found')
    else:
        print('dt_customer not datetime; cannot filter for recent complaints')

# %%
# Summary and recommended next steps (as comments)
"""
Summary (to include in notebook output):
- Completed data import, cleaning, and imputation of income by education+marital status median.
- Engineered features: age, total_children, total_spend, total_purchases.
- Performed winsorization on high-variance numeric features.
- Ran hypothesis tests where relevant columns existed; if required columns are missing the notebook explains how to add them.

Recommended next steps:
1. Review categorical value mappings for education and marital status; harmonize typos & variants.
2. If wanting stricter outlier handling, consider robust scalers or transformation (log) for spend/income.
3. Build predictive models (e.g., logistic regression) for campaign acceptance and customer churn.
4. Create a dashboard (Plotly Dash, Streamlit or Tableau) for the Head of S&M with time-sliced filters.

Notes:
- This script uses heuristics to detect columns; adjust variable names in the script to exactly match your dataset column names for best results.
- Drop-in: replace DATA_PATH at top if file is in another directory.
"""

print('\nNotebook script complete.\n')
