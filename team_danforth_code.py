# EDA

# 1. Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_demo   = pd.read_csv('MMA 869 Team Project/traindemographics.csv')   # Customer demographic information
df_perf   = pd.read_csv('MMA 869 Team Project/trainperf.csv')           # Current loan performance
df_prev   = pd.read_csv('MMA 869 Team Project/trainprevloans.csv')      # Historical loan records


print(df_demo.shape, df_perf.shape, df_prev.shape)

# 2. Data Structure & Overview of Missing Values
# Data type & Non-null count
print(df_demo.info())
print(df_perf.info())
print(df_prev.info())

# Proportion of missing values
for name, df in [('demo', df_demo), ('perf', df_perf), ('prev', df_prev)]:
    miss_pct = df.isnull().mean().sort_values(ascending=False)
    print(f"\n== {name} missing % ==\n", miss_pct[miss_pct>0].head(10))

# 3. Distribution of the target variable
# good_bad_flag: 1=good, 0=bad
print(df_perf['good_bad_flag'].value_counts(normalize=True))
df_perf['good_bad_flag'].value_counts().plot.bar()
plt.title('Target Distribution')
plt.xlabel('good_bad_flag')
plt.ylabel('Count')
plt.show()

# 4. Exploration of Numerical Features
num_cols = ['loanamount', 'totaldue', 'termdays']
# Descriptive statistics
print(df_perf[num_cols].describe())

# histogram
for col in num_cols:
    plt.figure()
    df_perf[col].hist(bins=50)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 5. Exploration of Categorical Features
cat_cols = ['bank_account_type', 'employment_status_clients', 'level_of_education_clients']
for col in cat_cols:
    vc = df_demo[col].value_counts(dropna=False)
    print(col, vc.head(10))
    vc.plot.bar()
    plt.title(col)
    plt.show()

# 6. The correlation between characteristics and goals

# 6.1. Map the target label to numeric
# Assume df_perf['good_bad_flag'] was originally 'Good' / 'Bad'
df_perf['good_bad_flag'] = df_perf['good_bad_flag'].map({'Good': 1, 'Bad': 0})
# Verify the mapping
print(df_perf['good_bad_flag'].dtype)
print(df_perf['good_bad_flag'].value_counts())

# 6.2. Numerical features vs. target: Box plots

num_cols = ['loanamount', 'totaldue', 'termdays']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    # Draw a box plot of each numeric column grouped by the default flag
    df_perf.boxplot(column=col, by='good_bad_flag')
    plt.title(f"{col} vs Default Flag")
    plt.suptitle("")               # Remove the automatic subtitle
    plt.xlabel("good_bad_flag (1 = Good, 0 = Bad)")
    plt.ylabel(col)
    plt.show()

# 6.3. Categorical features vs. default rate
# Merge demographics and performance tables
df_main = df_perf.merge(df_demo, on='customerid', how='left')

# List the categorical columns
cat_cols = ['bank_account_type', 'employment_status_clients', 'bank_name_clients']

for col in cat_cols:
    # Compute the average repayment rate per category (0 = higher default rate, 1 = fully repaid)
    grp = (
        df_main
        .groupby(col)['good_bad_flag']
        .mean()
        .sort_values()
    )
    print(f"\nDefault Rate by {col}:\n", grp)

    # Plot the repayment rate by category
    plt.figure(figsize=(8, 4))
    grp.plot.bar()
    plt.title(f"Average Repayment Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Repayment Rate (1 = all Good)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# 7. Aggregation
agg_prev = df_prev.groupby('customerid').agg(
    prev_loan_count=('systemloanid', 'nunique'),
    prev_loan_amt_mean=('loanamount', 'mean'),
    prev_term_mean=('termdays', 'mean'),
    prev_delay_mean=('firstrepaiddate', lambda d:
        ((pd.to_datetime(d) - pd.to_datetime(df_prev['creationdate'])).dt.days).mean()),
    prev_early_payment_mean=("firstduedate", lambda d:
        ((pd.to_datetime(d) - pd.to_datetime(df_prev["firstrepaiddate"])).dt.days).mean()),
).reset_index()

df_main = (df_perf
           .merge(df_demo, on='customerid', how='left')
           .merge(agg_prev, on='customerid', how='left'))

print(df_main.shape)
print(df_main.head())

# 8. Heatmap
import numpy as np
import matplotlib.pyplot as plt

# Calculate the correlation matrix
num_cols = df_main.select_dtypes(include=[np.number]).columns
corr = df_main[num_cols].corr().values

# Draw
plt.figure(figsize=(12,10))
im = plt.imshow(corr, cmap='viridis', interpolation='none')
plt.colorbar(im, fraction=0.046, pad=0.04)

# Axis labels
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols, rotation=0)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Data Preprocessing

# duplication
duplicates_main = df_main[df_main['customerid'].duplicated()]
print(f"duplicated customerid: {duplicates_main.shape[0]}")
print("duplication", duplicates_main['customerid'].unique())

df_main.drop_duplicates(subset='customerid', inplace=True)
df_main.info()

# date manipulation, since most model cannot handle date values
df_main['birthdate'] = pd.to_datetime(df_main['birthdate'], errors='coerce')
df_main['creationdate'] = pd.to_datetime(df_main['creationdate'], errors='coerce')
df_main['approveddate'] = pd.to_datetime(df_main['approveddate'], errors='coerce')

df_main['age'] = 2025 - df_main['birthdate'].dt.year
df_main['approval_delay'] = (df_main['approveddate'] - df_main['creationdate']).dt.total_seconds() / 86400
df_main.info()

# Non-Tree Models

# Target variable
y_train = df_main['good_bad_flag']
# features
drop_cols=['customerid', 'systemloanid', 'birthdate', 'creationdate', 'approveddate', 'good_bad_flag']
x_train = df_main.drop(columns = drop_cols)
x_train.info()
x_train.head()

y_train.info()
y_train.head()

# Data Processing

# Missing values
num_cols = x_train.select_dtypes(include=['float64', 'int64']).columns
cat_cols = x_train.select_dtypes(include='object').columns
# numerical values
x_train[num_cols] = x_train[num_cols].fillna(x_train[num_cols].median())
# categorical values
x_train[cat_cols] = x_train[cat_cols].fillna('missing')

# Logistic Regression

# Feature Engineering

# Encoding
# one-hot encoding
x_train = pd.get_dummies(x_train, columns=cat_cols, drop_first=True)

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Models

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_model = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(log_model, x_train, y_train, cv=10, scoring='accuracy')

print(f"Logistic Regression Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# KNN

# Feature Engineering

# Encoding
# one-hot encoding
x_train = pd.get_dummies(x_train, columns=cat_cols)

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Model

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn_model, x_train, y_train, cv=10, scoring='accuracy')

print(f"KNN Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_neighbors': list(range(3, 31, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


knn_model = KNeighborsClassifier()


rand_search = RandomizedSearchCV(
    estimator=knn_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=10,
    scoring='accuracy',
    random_state=42,
    verbose=1,
    n_jobs=-1
)


rand_search.fit(x_train, y_train)


print("Best Accuracy:", rand_search.best_score_) # Best Accuracy: 0.789608044842861
print("Best Params:", rand_search.best_params_)

# Tree Models

# Target variable
y_train = df_main['good_bad_flag']

# features
drop_cols=['customerid', 'systemloanid', 'birthdate', 'creationdate', 'approveddate', 'good_bad_flag']
x_train = df_main.drop(columns = drop_cols)

y_train.info()
y_train.head()

# LightGBM

# Ensure all categorical columns are string type so they can be marked as categorical
for col in x_train.select_dtypes(include='object').columns:
    x_train[col] = x_train[col].astype('category')

from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import numpy as np

# Instantiate unoptimized LGBM model
lgbm_base = LGBMClassifier(random_state=42)

# Cross-validation using accuracy
scores = cross_val_score(lgbm_base, x_train, y_train, cv=10, scoring='accuracy')

# Baseline score
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})") # Accuracy: 0.7889 (+/- 0.0139)

# Feature Engineering ( doesn't work, lower the performance)

# try FE
# details of approval -> lower the accury to 0.7880
x_train['approved_month'] = df_main['approveddate'].dt.month
x_train['approved_day'] = df_main['approveddate'].dt.day
x_train['approved_dayofweek'] = df_main['approveddate'].dt.dayofweek
x_train.head()

# Instantiate unoptimized LGBM model
lgbm_fe = LGBMClassifier(random_state=42)

# Cross-validation using accuracy
scores = cross_val_score(lgbm_fe, x_train, y_train, cv=10, scoring='accuracy')

# fe score
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")  # Accuracy: 0.7886 (+/- 0.0084), the same as baseline model

# feature importance
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

lgbm_fe.fit(x_train, y_train)

# top-20 features
lgb.plot_importance(lgbm_fe, max_num_features=20, importance_type='gain', figsize=(10, 6))
plt.title("Top 20 Feature Importances (Gain)")
plt.show()

# full table
feat_imp = pd.DataFrame({
    'feature': x_train.columns,
    'importance': lgbm_fe.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feat_imp.head(10))

# interaction fe -> further lower to 0.7873
# Combine education level and employment status - doesn't work
# x_train['edu_emp_status'] = x_train['level_of_education_clients'].astype(str) + '_' + x_train['employment_status_clients'].astype(str)

# Behaviour patterns
x_train['acctype_weekday'] = x_train['bank_account_type'].astype(str) + '_' + df_main['approveddate'].dt.dayofweek.astype(str)

# Regional risk patterns. - doesn't work
# x_train['geo_delay'] = (x_train['latitude_gps'] + x_train['longitude_gps']) * x_train['prev_delay_mean']

print(x_train.columns)

for col in x_train.select_dtypes(include='object').columns:
    x_train[col] = x_train[col].astype('category')

# Instantiate unoptimized LGBM model
lgbm_interac = LGBMClassifier(random_state=42)

# Cross-validation using accuracy
scores = cross_val_score(lgbm_interac, x_train, y_train, cv=10, scoring='accuracy')

# interac score
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")  # Accuracy: 0.7898 (+/- 0.0112)

# feature importance
lgbm_interac.fit(x_train, y_train)

# top-20 features
lgb.plot_importance(lgbm_interac, max_num_features=20, importance_type='gain', figsize=(10, 6))
plt.title("Top 20 Feature Importances (Gain)")
plt.show()

# full table
feat_imp = pd.DataFrame({
    'feature': x_train.columns,
    'importance': lgbm_interac.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feat_imp.head(10))

# Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

# 1. Define hyperparameter grid
param_dist = {
    'num_leaves': [15, 31, 63],
    'max_depth': [-1, 5, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
}

# 2. Initialize model
lgbm_model = LGBMClassifier(random_state=42, class_weight='balanced')

# 3. RandomizedSearchCV
rand_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_dist,
    n_iter=100,  # number of random combinations to try
    cv=10,
    scoring='accuracy',
    verbose=1,
)

# 4. Fit
rand_search.fit(x_train, y_train)

# 5. Best parameters & score
print("Best Accuracy: ", rand_search.best_score_) #Best Accuracy:  0.7644065494289626
print("Best Params: ", rand_search.best_params_)

# import optuna

# # Step 1: Define objective function for Optuna
# def objective(trial):
#     params = {
#         'num_leaves': trial.suggest_int('num_leaves', 20, 128),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 500),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
#         'random_state': 42,
#         'class_weight': 'balanced'
#     }

#     model = LGBMClassifier(**params)

#     # Use cross-validation to evaluate accuracy
#     scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')

#     return scores.mean()

# # Step 2: Start Optuna optimization
# study = optuna.create_study(direction='maximize')  # maximize accuracy
# study.optimize(objective, n_trials=50, timeout=900)  # try 50 sets or stop after 15 mins

# print("Best Accuracy: ", study.best_value)
# print("Best Parameters: ", study.best_params)

# Catboost

# features
drop_cols=['customerid', 'systemloanid', 'birthdate', 'creationdate', 'approveddate', 'good_bad_flag']
x_train = df_main.drop(columns = drop_cols)
x_train.info()
x_train.head()

# Feature Engineering

# details of approval -> increased to 0.8
x_train['approved_month'] = df_main['approveddate'].dt.month
x_train['approved_day'] = df_main['approveddate'].dt.day
x_train['approved_dayofweek'] = df_main['approveddate'].dt.dayofweek

# Missing Values, catboost cannot handle categorical missing value
for col in x_train.select_dtypes(include='object').columns:
    x_train[col] = x_train[col].fillna('missing')

# Missing Numerical Values - DOESNT WORK: accuracy goes down -> 0.7958
# cols_to_fill_0 = ['prev_loan_count', 'prev_loan_amt_mean', 'prev_term_mean', 'prev_delay_mean']
# x_train[cols_to_fill_0] = x_train[cols_to_fill_0].fillna(0)

# interaction on categorical feature
# edu x employment ->0.7967
# x_train["edu_employment"] = x_train["level_of_education_clients"].astype(str) + "_" + x_train["employment_status_clients"].astype(str)
# acc type x approvedate
# x_train["acctype_weekday"] = x_train["bank_account_type"].astype(str) + "_" + df_main["approveddate"].dt.dayofweek.astype(str)


x_train.info()
x_train.head(20)

# Models

!pip install catboost

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score


# Identify categorical columns
cat_features = x_train.select_dtypes(include='object').columns.tolist()

#  Instantiate CatBoost model (no tuning)
catboost_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    cat_features=cat_features
)

# Cross-validation
scores = cross_val_score(catboost_model, x_train, y_train, cv=10, scoring='accuracy', error_score='raise')

# Print results
print(f"CatBoost Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})") # Accuracy: 0.7969 (+/- 0.0099)

#Baseline Accurracy: 0.7965 (+/- 0.0089) - without further fe and tuning
#Fe-details on approveddate: Accuracy: 0.7992 (+/- 0.0124)
#log_transformation lower the performance; Accuracy: 0.7974 (+/- 0.0119)
#drop least important features lower the performance to 0.7979 (+/- 0.0123)

# feature importance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Identify categorical columns
cat_features = x_train.select_dtypes(include='object').columns.tolist()

# Train CatBoost model explicitly on full training set
catboost_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    cat_features=cat_features,
)
catboost_model.fit(x_train, y_train)

# Now get feature importances
cat_feat_importance = catboost_model.get_feature_importance()
cat_feat_names = x_train.columns

# Build dataframe
cat_feat_imp_df = pd.DataFrame({
    'feature': cat_feat_names,
    'importance': cat_feat_importance
}).sort_values(by='importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=cat_feat_imp_df.head(20), palette='viridis')
plt.title('Top 20 Feature Importances (CatBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# print(cat_feat_imp_df.head(5))
print(cat_feat_imp_df.tail(5))

# Skewness in top features

import seaborn as sns
import matplotlib.pyplot as plt

for col in ['prev_early_payment_mean', 'prev_delay_mean', 'approved_day', 'prev_term_mean']:
    plt.figure(figsize=(6, 4))
    sns.histplot(x_train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# fe: log_transformation - doesnt work， lower the performance

# log-transformation for top features
x_train['prev_delay_mean_log'] = np.log1p(x_train['prev_delay_mean'])
x_train['prev_early_payment_mean_log'] = np.log1p(x_train['prev_early_payment_mean'])
# x_train['approved_day_log'] = np.log1p(x_train['approved_day'])
x_train['prev_term_mean_log'] = np.log1p(x_train['prev_term_mean'])

x_train.info()

# try model
# Identify categorical columns
cat_features = x_train.select_dtypes(include='object').columns.tolist()

#  Instantiate CatBoost model (no tuning)
catboost_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    cat_features=cat_features
)

# Cross-validation
scores = cross_val_score(catboost_model, x_train, y_train, cv=10, scoring='accuracy', error_score='raise')

# Print results
print(f"CatBoost Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Drop least important features - doesn't work， performance decreases

print(cat_feat_imp_df.tail(5))

low_importance_features = ['bank_branch_clients']
x_train = x_train.drop(columns=low_importance_features)
x_train.info()

# try model
# Identify categorical columns
cat_features = x_train.select_dtypes(include='object').columns.tolist()

#  Instantiate CatBoost model (no tuning)
catboost_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    cat_features=cat_features
)

# Cross-validation
scores = cross_val_score(catboost_model, x_train, y_train, cv=10, scoring='accuracy', error_score='raise')

# Print results
print(f"CatBoost Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})") # Accuracy:  0.7972 (+/- 0.0102) goes down

# Hyperparameter Tuning

from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

#  Define hyperparameter search space
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.1],
    'iterations': [100, 300, 500],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.1, 0.5, 1.0],
    'random_strength': [1, 5, 10],
    'border_count': [32, 64, 128],
    'rsm': [0.8, 1.0],  # feature sub-sampling
}

#  Instantiate CatBoostClassifier
catboost_model = CatBoostClassifier(
    cat_features=cat_features,
    verbose=0,
    random_state=42
)

#  Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=10,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit
random_search.fit(x_train, y_train)

# Print results
print("Best Accuracy: ", random_search.best_score_) # Best Accuracy:  0.8005998992295258
print("Best Parameters: ", random_search.best_params_)
# Best Parameters:
# {'rsm': 1.0, 'random_strength': 1, 'learning_rate': 0.03, 'l2_leaf_reg': 9, 'iterations': 500, 'depth': 4, 'border_count': 32, 'bagging_temperature': 1.0}

# optuna

!pip install optuna

# compute class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)

class_weights_dict = dict(zip(np.unique(y_train), class_weights))

import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

# define objective function
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
        "rsm": trial.suggest_float("rsm", 0.7, 1.0),
        "cat_features": cat_features,
        "verbose": 0,
        "random_state": 42,
        "class_weights": class_weights_dict  # handle class imbalance
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
    return score.mean()

# tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best Accuracy: ", study.best_value)
print("Best Hyperparameters: ", study.best_params)

# Prediction

# best model: catboost
from catboost import CatBoostClassifier, Pool
cat_features = x_train.select_dtypes(include='object').columns.tolist()
best_params = {
    'rsm': 0.8,
    'random_strength': 10,
    'learning_rate': 0.03,
    'l2_leaf_reg': 7,
    'iterations': 500,
    'depth': 8,
    'border_count': 64,
    'bagging_temperature': 0.1
}
final_catboost_model = CatBoostClassifier(
    **best_params,
    cat_features=cat_features,
    random_state=42,
    verbose=0
)
final_catboost_model.fit(x_train, y_train)

# process test dataset
df_test_demo = pd.read_csv('MMA 869 Team Project/testdemographics.csv')
df_test_prev = pd.read_csv('MMA 869 Team Project/testprevloans.csv')
df_test_perf = pd.read_csv('MMA 869 Team Project/testperf.csv')

# 1. remove duplication
# testdemographics
duplicates_demo = df_test_demo[df_test_demo['customerid'].duplicated()]
print(f"testdemographics duplicated customerid: {duplicates_demo.shape[0]}")

# testperf
duplicates_perf = df_test_perf[df_test_perf['customerid'].duplicated()]
print(f"testperf duplicated customerid: {duplicates_perf.shape[0]}")
print("testdemographics:", duplicates_demo['customerid'].unique())
# remove duplication in demographic data
df_test_demo.drop_duplicates(subset='customerid', inplace=True)

# 2. aggregation
agg_test_prev = df_test_prev.groupby('customerid').agg(
    prev_loan_count=('systemloanid', 'nunique'),
    prev_loan_amt_mean=('loanamount', 'mean'),
    prev_term_mean=('termdays', 'mean'),
    prev_delay_mean=('firstrepaiddate', lambda d:
        ((pd.to_datetime(d) - pd.to_datetime(df_test_prev['creationdate'])).dt.days).mean())
).reset_index()

df_test_main = (df_test_perf
           .merge(df_test_demo, on='customerid', how='left')
           .merge(agg_test_prev, on='customerid', how='left'))
print(df_test_main.info())

# 3. date manipulation, since most model cannot handle date values
df_test_main['birthdate'] = pd.to_datetime(df_test_main['birthdate'], errors='coerce')
df_test_main['creationdate'] = pd.to_datetime(df_test_main['creationdate'], errors='coerce')
df_test_main['approveddate'] = pd.to_datetime(df_test_main['approveddate'], errors='coerce')


df_test_main['age'] = 2025 - df_test_main['birthdate'].dt.year
df_test_main['approval_delay'] = (df_test_main['approveddate'] - df_test_main['creationdate']).dt.total_seconds() / 86400
df_test_main.info()

# 4. features
# features
drop_cols=['customerid', 'systemloanid', 'birthdate', 'creationdate', 'approveddate']
x_test = df_test_main.drop(columns = drop_cols)

# details of approval
x_test['approved_month'] = df_test_main['approveddate'].dt.month
x_test['approved_day'] = df_test_main['approveddate'].dt.day
x_test['approved_dayofweek'] = df_test_main['approveddate'].dt.dayofweek

# Missing Values, catboost cannot handle categorical missing value
for col in x_test.select_dtypes(include='object').columns:
    x_test[col] = x_test[col].fillna('missing')

x_test.info()
x_test.head()

#prediction
y_pred = final_catboost_model.predict(x_test)
customer_ids = df_test_main['customerid']
submission = pd.DataFrame({
    'customerid': customer_ids,
    'prediction': y_pred
})
submission.info()
submission.head()
submission.to_csv('submission.csv', index=False)
