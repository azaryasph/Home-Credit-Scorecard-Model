# %% [markdown]
# # FINAL TASK PBI HOME CREDIT INDONESIA
# 
# <img src="https://www.homecredit.co.id/_next/static/images/og-image-49523962f23f5e3dcc2bf6dc246189bd.png" width="500" height="300">
# 
# ## Project Problem Statement
# Home Credit Indonesia wants to minimize risk of losing client that potentially will be able to pay the loan. The company wants to predict the probability of a client to be able to pay the loan. The company also wants to know the factors that influence the probability of a client to be able to pay the loan.
# 
# ## Project Goals
# Increase apporval rate of good client at least 5% and decrease approval rate of bad client at least 5%.
# 
# ## Project Objectives
# 1. Build a model to predict the probability of a client to be able to pay the loan.
# 2. Identify the factors that influence the probability of a client to be able to pay the loan.
# 
# ## Project Plan
# 1. Data Preparation
# 2. Exploratory Data Analysis
# 3. Data Preprocessing
# 4. Model Building
# 5. Model Evaluation
# 6. Model Interpretation

# %% [markdown]
# ## Import Library

# %%
import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import modules.missing as ms
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split
import toad
from toad.metrics import KS, AUC
from toad.plot import bin_plot, badrate_plot

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
import shap

# %% [markdown]
# ## Load Cleaned Data

# %%
app_train = pd.read_csv('../data/application_train_cleaned.csv')
bb = pd.read_csv('../data/bureau_balance.csv')
bureau = pd.read_csv('../data/bureau_clean.csv')
ccb = pd.read_csv('../data/credit_card_balance_clean.csv')
installments = pd.read_csv('../data/installments_payments_clean.csv')
pos_cash = pd.read_csv('../data/pos_cash_balance_clean.csv')
previous = pd.read_csv('../data/previous_application_clean1.csv')

# %% [markdown]
# ## Join Data

# %% [markdown]
# ### Bureau data groupby and joining

# %%
# aggreagate bureau balance Months' balance with mean for every unique bureau id
bb_agg = bb.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].mean().reset_index(name='AVG_MONTHS_BALANCE')

# join aggregated bureau balance with bureau
bureau_semi_join = pd.merge(bureau, bb_agg, on='SK_ID_BUREAU', how='left')

# aggregate current credit amount and current debt(on credit bureau) with sum for every unique current application id
bureau_agg = bureau_semi_join.groupby('SK_ID_CURR')[['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT']].sum().reset_index()

# %% [markdown]
# ### Join bureau and application train

# %%
# left join application train with aggregated bureau 
train_bureau = pd.merge(app_train, bureau_agg, on='SK_ID_CURR', how='left')

# %% [markdown]
# ### Payment installments aggregation

# %%
# installment payment data joining and aggregating
installments_agg = installments.groupby(['SK_ID_PREV', 'SK_ID_CURR'])[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum().reset_index()

# calculating difference between installment payment and installment amount
installments_agg['DIFF_INSTALLMENT_PAYMENT'] = installments_agg['AMT_INSTALMENT'] - installments_agg['AMT_PAYMENT']
installments_agg = installments_agg.drop(columns='SK_ID_CURR', axis=1)

# %% [markdown]
# ### Join payment isntallments with previous application

# %%
# join installment with previous application
prev_installments = pd.merge(previous, installments_agg, on='SK_ID_PREV', how='left')

# %% [markdown]
# ### Credit Card Balance Aggregate and Join

# %%
ccb_agg = ccb.groupby('SK_ID_PREV')[['AMT_BALANCE', 'AMT_PAYMENT_TOTAL_CURRENT']].sum().reset_index()

prev_ccb = pd.merge(prev_installments, ccb_agg, on='SK_ID_PREV', how='left')

# %% [markdown]
# ### Pos Cash Balance Aggregate and Join

# %%
pc_agg = pos_cash.groupby('SK_ID_PREV')['SK_DPD'].mean().reset_index(name='AVERAGE_DPD')

prev_pc = pd.merge(prev_ccb, pc_agg, on='SK_ID_PREV', how='left')

# %% [markdown]
# ### Train Full

# %%
df_train_full = pd.merge(train_bureau, prev_pc, on='SK_ID_CURR', how='left', suffixes=('_curr', '_prev'))

# %%
# train_full.to_csv('../data/train_full.csv', index=False)

# %% [markdown]
# ## Preliminary Data Analysis

# %%
pd.set_option('display.max_columns', None)

display(df_train_full.sample(10))
print(f'Application train data contains: \n{df_train_full.shape[0]} rows and {df_train_full.shape[1]} columns')
print(f'\nApplication train data information:\n')
print(df_train_full.info())
print(f'\nStatistical Summary of numerical columns:\n')

numerical = df_train_full.select_dtypes(include='number')
display(numerical.describe())

categorical = df_train_full.select_dtypes(include='object')
display(categorical.describe())

# %%
xna_goods = df_train_full.NAME_GOODS_CATEGORY.value_counts()[0]
total_data = df_train_full.shape[0]
print(f'Proportion of XNA values in NAME_GOODS_CATEGORY {round((xna_goods / total_data) * 100, 2)}%')

xna_client = df_train_full.NAME_CLIENT_TYPE.value_counts()[0]
print(f'Proportion of XNA values in NAME_CLIENT_TYPE: {round((xna_client / total_data) * 100, 2)}%')


# %% [markdown]
# for XNA wich is not not available, the value proportion is half of the data, that we have to drop this because rest of the value around 44% is too small and considered not valid. 

# %% [markdown]
# ### Fix / Replace invalid values in the data

# %%
train_full = df_train_full.copy()

# x train categorical value binning
train_full['AGE'] = round(abs(train_full['DAYS_BIRTH'] / 365.25)).astype(int)

# group age by twenty and so on
age_group = [
    (train_full['AGE']>20) & (train_full['AGE']<=30),
    (train_full['AGE']>30) & (train_full['AGE']<=40),
    (train_full['AGE']>40) & (train_full['AGE']<=50),
    (train_full['AGE']>50) & (train_full['AGE']<=60),
    (train_full['AGE']>60) & (train_full['AGE']<=70),
]
age_lab = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty']

# Create the AGE_GROUP column with ordered categories
train_full['AGE_GROUP'] = pd.Categorical(np.select(age_group, age_lab), categories=age_lab, ordered=True)


train_full = train_full[train_full.NAME_FAMILY_STATUS != 'Unknown']

# make a function to replace columns value contains XNA or XAP with np.nan
def replace_xna_xap(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['XNA', 'XAP'], np.nan)
    return df

train_full = replace_xna_xap(train_full)

# replace abnormal value
replace_val = {4.5 : 5,
               0.5 : 1}
train_full['CNT_FAM_MEMBERS'] = train_full['CNT_FAM_MEMBERS'].replace(replace_val)

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Missing values Checking

# %%
ms.plot_missing_percentage(train_full)

# %% [markdown]
# Our data Contains a lot of missing values,
# - For the columns that have missing values more than 90% of the data, i will drop them.
# - AVERAGE DPD missing values seems reasonable, that there's client that didn't have any DPD, or didn't have any previous application.
# - for the columns `DIFF_INSTALMENT_PAYMENT`, `AMT_PAYMENT` and `AMT_INSTALMENT` i will fill the missing values with 0, because it's reasonable to assume that the client didn't have any previous application too.
# - the rest of it i will leave it be, so when there's missing values, while we do woe binning, it will be treated as a separate category.

# %% [markdown]
# ### Oulier Cheking

# %%
numvisual = train_full[[
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT_curr', 'AMT_ANNUITY_curr',
    'AMT_GOODS_PRICE_curr','REGION_POPULATION_RELATIVE','DAYS_BIRTH',
    'DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START_curr',
    'EXT_SOURCE_2','OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
    'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_ANNUITY_prev', 'AMT_APPLICATION',
    'AMT_CREDIT_prev','AMT_GOODS_PRICE_prev', 'HOUR_APPR_PROCESS_START_prev',
    'NFLAG_LAST_APPL_IN_DAY', 'DAYS_DECISION', 'SELLERPLACE_AREA',
    'CNT_PAYMENT', 'DAYS_LAST_DUE', 'AMT_INSTALMENT', 'AMT_PAYMENT',
    'DIFF_INSTALLMENT_PAYMENT', 'AMT_BALANCE', 'AMT_PAYMENT_TOTAL_CURRENT',
    'AVERAGE_DPD'
          ]].copy()

colgroup = [numvisual.columns[n:n+4] for n in range(0, len(numvisual.columns), 4)]

# Create a 9x4 grid of subplots
fig, axs = plt.subplots(4, 10, figsize=(30, 15))

# for each group of 4 columns
for i, group in enumerate(colgroup):
    # For eachg column in the group
    for j, column in enumerate(group):
        # Create a boxplot for the apporopriate subplot
        sns.boxplot(y=numvisual[column], ax=axs[j, i ], color='blue', linewidth=1.5, orient='v')
        axs[j, i].set_title(column)

# show plot
plt.tight_layout()
plt.show()

# %% [markdown]
# From boxplot above we can see that there's outliers in our numerical columns

# %%
counts = train_full['TARGET'].value_counts(normalize=True)
fig, ax = plt.subplots(figsize=(15, 5))
fig.patch.set_facecolor('#FFFFFF')

gbp = ['#69ba2d', '#d9000c']

ax.barh(['Clients'], counts[0], color=gbp[0], label='Good Clients')
ax.barh(['Clients'], counts[1], left=counts[0], color=gbp[1], label='Bad Clients')
ax.set_facecolor('#FFFFFF')
sns.despine()

for i, v in enumerate(counts):
    ax.text(v/2 if i == 0 else counts[0] + v/2, 0, f'{v:.1%}', color='w', fontweight='bold', ha='center', va='center', fontsize=16)

# plt.title('Proportion of Good and Bad Clients')
# plt.text(0.5, 0.5, f'Good Clients: {counts[0]:.1%}\nBad Clients: {counts[1]:.1%}', fontsize=14, ha='center', va='center', fontweight='bold')
plt.text(0.4, 0.59, f"THERE'S MORE GOOD CLIENTS THAN BAD CLIENTS IN OUR DATA", fontsize=20, ha='center', va='center', fontweight='bold')
plt.text(-0.001, 0.425, f"However, HCI encounters obstacles for consumers who incur losses due to bad Debt.\nApproval for consumers with bad debt characteristics must be reduced to reduce the company's bad debt.", fontsize=18.5)
plt.grid(False)
plt.yticks(['Clients'], fontsize=14)
plt.xticks([])
legend = plt.legend(bbox_to_anchor=(1.1, 0.90), fontsize=12)
frame = legend.get_frame()
frame.set_facecolor('#FFFFFF')
plt.show()
print(f"There's {train_full.shape[0]} Clients in our data")

# %% [markdown]
# From the plot above, we can see that:
# - From total 1430155 clients in our data.
# - 1,306,815 (91.4%) clients don't have payment difficulties. 
# - 123,340 (8.6%) clients have payments difficulties.
# - The data is highly imbalanced.

# %%
# Get the top 4 income types
top_4_income_types = train_full['NAME_INCOME_TYPE'].value_counts().index[:4]

# Filter the DataFrame to include only the top 4 income types
filtered_train_full = train_full[train_full['NAME_INCOME_TYPE'].isin(top_4_income_types)]

# Calculate the order of levels by value counts
order = filtered_train_full['NAME_INCOME_TYPE'].value_counts().index

fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('#FFFFFF')

# First subplot
ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 cols, subplot 1
ax1.set_facecolor('#FFFFFF')
sns.countplot(x='NAME_INCOME_TYPE', data=filtered_train_full, palette='Set1', ax=ax1, order=order, edgecolor='black')
plt.grid(False)
plt.title('Count of Clients by Income Type', fontsize=20, fontweight='bold', y=1.05)
plt.xlabel(' ')
plt.ylabel('Count')
plt.xticks(rotation=0, fontsize=17)

ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 cols, subplot 2
ax2.set_facecolor('#FFFFFF')
gbp = ['#69ba2d', '#d9000c']
income_target = filtered_train_full.groupby('NAME_INCOME_TYPE')['TARGET'].value_counts(normalize=True).unstack()

# Calculate value counts of 'NAME_INCOME_TYPE'
income_counts = filtered_train_full['NAME_INCOME_TYPE'].value_counts()

# Sort 'income_target' by 'income_counts'
income_target = income_target.loc[income_counts.index]

income_target.plot(kind='bar', stacked=True, color=gbp, ax=ax2, edgecolor='black')

# Add annotations
for p in ax2.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax2.annotate(f'{height:.0%}', (x + width/2, y + height/2), ha='center')

sns.despine()
plt.title('Proportion of Good and Bad Clients by Income Type', fontsize=20, fontweight='bold', y=1.05)
plt.xlabel(' ')
plt.ylabel('Proportion')
plt.xticks(rotation=0, fontsize=17)
plt.grid(False)
plt.legend(title='Client', bbox_to_anchor=(1.05, 1), fontsize=14, labels=['Good', 'Bad'])

plt.tight_layout()
plt.show()

# %% [markdown]
# From the plot above we can infer that:
# - Focus giving loan to clients that status income type is just working, because they are the most frequent client, and their the approval rate from this income type is not less than 90% wich is just fine.
# - Consider giving loan to client that the income type are from Commercial associate, because they are the second most frequent client, and their the approval rate is more than client that just working.

# %% [markdown]
# ### Client's Age default rate

# %%
fig = plt.figure(figsize=(20, 8))

ax1 = plt.subplot(1, 2, 1)
ax1.set_facecolor('#FFFFFF')
sns.countplot(x='AGE_GROUP', data=train_full, palette='Set1', ax=ax1, edgecolor='black')
plt.grid(False)
plt.title('Count of Clients by Age Group', fontsize=20, fontweight='bold', y=1.05)
plt.xlabel(' ')
plt.ylabel('Count')
plt.xticks(rotation=0, fontsize=14)

ax2 = plt.subplot(1, 2, 2)
ax2.set_facecolor('#FFFFFF')
gbp = ['#69ba2d', '#d9000c']

age_target = train_full.groupby('AGE_GROUP')['TARGET'].value_counts(normalize=True).unstack()

# Sort the DataFrame by the index
age_target = age_target.sort_index()

age_target.plot(kind='bar', stacked=True, color=gbp, ax=ax2, edgecolor='black')

for p in ax2.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax2.annotate(f'{height:.0%}', (x + width/2, y + height/2), ha='center')

sns.despine()
plt.title('Proportion of Good and Bad Clients by Age Group', fontsize=20, fontweight='bold', y=1.05)
plt.xlabel(' ')
plt.ylabel('Proportion')
plt.xticks(rotation=0, fontsize=14)
plt.grid(False)
plt.legend(title='Client', bbox_to_anchor=(1.05, 1), fontsize=14, labels=['Good', 'Bad'])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Train Test Split

# %%
X = train_full.drop(columns='TARGET', axis=1)
y = train_full[['TARGET']]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1103)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% [markdown]
# ## Data filtering

# %%
toad.quality(X_train, target=y_train['TARGET'], iv_only=True)

# %%
Xtrain_selected, drop_list = toad.selection.select(frame=X_train,
                                                  target=y_train['TARGET'],
                                                  empty=0.5,
                                                  iv=0.02,
                                                  corr=0.7,
                                                  return_drop=True)

print("keep:",Xtrain_selected.shape[1],
      "drop empty:",len(drop_list['empty']),
      "drop iv:",len(drop_list['iv']),
      "drop corr:",len(drop_list['corr']))

display(Xtrain_selected)
print(f'\n{drop_list}')

# %%
selected_features = Xtrain_selected.columns

X_test = X_test[selected_features]

# %%
# output the iv table to a dataframe
def iv_importance(data_selected, label):
    iv_import_feat = toad.quality(data_selected, label, iv_only=True)
    iv_import_feat = iv_import_feat['iv']
    iv_import_feat = iv_import_feat.reset_index()
    iv_import_feat.columns = ['name', 'iv']
    return iv_import_feat

df_iv = iv_importance(Xtrain_selected, y_train['TARGET'])

df_iv.sort_values(by='iv', ascending=False)

# %% [markdown]
# ## Handling Outlier

# %%
def cap_outliers(data, columns):
    # Create a copy of the data to avoid modifying the original DataFrame
    result = data.copy()

    # Loop over each column in the provided list
    for col in columns:
        # Calculate the first quartile (Q1) for the current column
        Q1 = result[col].quantile(0.25)
        # Calculate the third quartile (Q3) for the current column
        Q3 = result[col].quantile(0.75)
        # Calculate the Interquartile Range (IQR) for the current column
        IQR = Q3 - Q1
        # Calculate the lower bound for the current column
        lower_bound = Q1 - (1.5 * IQR)
        # Calculate the upper bound for the current column
        upper_bound = Q3 + (1.5 * IQR)
        # Replace values in the current column that are less than the lower bound with the lower bound
        result[col] = np.where(result[col] < lower_bound, lower_bound, result[col])
        # Replace values in the current column that are greater than the upper bound with the upper bound
        result[col] = np.where(result[col] > upper_bound, upper_bound, result[col])
    # Return the DataFrame with capped outliers
    return result


# Select the names of all numerical columns in the DataFrame X_train_c2
outliers = Xtrain_selected.select_dtypes(include='number').columns

# Call the cap_outliers function on X_train_c2, passing in the names of the numerical columns
# This will cap the outliers in these columns, and the result is stored in X_train_c3
Xtrain_selected = cap_outliers(Xtrain_selected, outliers)

# %% [markdown]
# ## Feature Binning

# %%
# combine x_train and y_train
train = pd.concat([Xtrain_selected, y_train], axis=1)
sample = train.sample(frac=0.25, random_state=1103)

c = toad.transform.Combiner()
c.fit(X=sample.drop('TARGET', axis=1), y=sample['TARGET'], method='dt', n_bins=None, empty_separate=True)

# %%
bins_output = c.export()
bins_output

# %% [markdown]
# ## Transform WoE

# %%
# intialize the WOE transformer
td = toad.transform.WOETransformer()

# Transform the data into WoE values
Xtrain_selected_woe = td.fit_transform(c.transform(Xtrain_selected), y_train['TARGET'])

# transform test set
Xtest_selected_woe = td.fit_transform(c.transform(X_test), y_test['TARGET'])

# Calculate the Information Value (IV) for each feature
iv = toad.quality(Xtrain_selected_woe, y_train['TARGET'], iv_only=True)

iv

# %% [markdown]
# ## Model building

# %%
def eval_model(model, X_train, y_train, X_test, y_test, thresshold):
    
    model.fit(X_train, y_train)

    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    # print auc score with roc auc score
    print('Train AUC:', roc_auc_score(y_train, pred_train))
    print('Test AUC:', roc_auc_score(y_test, pred_test))
    # print('Train AUC:', AUC(pred_train, y_train))
    # print('Test AUC:', AUC(pred_test, y_test))

    print('Train Recall:', recall_score(y_train, pred_train > thresshold))
    print('Test Recall:', recall_score(y_test, pred_test > thresshold))

    print('Train Precision:', precision_score(y_train, pred_train > thresshold))
    print('Test Precision:', precision_score(y_test, pred_test > thresshold))

    fig, ax = plt.subplots(figsize=(11, 5))
    
    fpr, tpr, _ = roc_curve(y_test, pred_test)
    roc_auc = roc_auc_score(y_test, pred_test)

    # Plot ROC Curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Calculate the precision-recall curve points
    precision, recall, _ = precision_recall_curve(y_test, pred_test)
    
    # Calculate the average precision score
    avg_precision = average_precision_score(y_test, pred_test)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall curve (Average Precision = %0.2f)' % avg_precision, color='orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# %%
lr = LogisticRegression(random_state=1103, class_weight='balanced')
eval_model(lr, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, thresshold=0.42)

# %%
# initialize the lightgbm classifier
lgbm = LGBMClassifier(n_estimators=100, random_state=1103, class_weight='balanced')

eval_model(lgbm, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, thresshold=0.42)

# %% [markdown]
# From 2 model (Logistic Regression and LightGBM), we can see that LightGBM model has better performance than Logistic Regression model, from the AUC score, and the Precision Score.

# %% [markdown]
# ## Hyperparameter Tuning

# %%
# hyperparameter tuning lightgbm
num_leaves = [int(x) for x in np.linspace(start=20, stop=150, num=10)]
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# Create the random grid
random_grid = {'num_leaves': num_leaves,
               'max_depth': max_depth}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
lgbm = LGBMClassifier(n_estimators=100, random_state=1103, class_weight='balanced')
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
lgbm_result = RandomizedSearchCV(estimator=lgbm, param_distributions=random_grid, n_iter=100, cv=3, random_state=1103)

# print the best parameters
lgbm_result.fit(Xtrain_selected_woe, y_train)
lgbm_result.best_params_

# %%
# logistic regression hyperparameter tuning
penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['liblinear', 'saga']

# Create the random grid
random_grid = {'penalty': penalty,
               'C': C,
               'solver': solver}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
lr = LogisticRegression(random_state=1103, class_weight='balanced')
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
lr_result = RandomizedSearchCV(estimator=lr, param_distributions=random_grid, n_iter=100, cv=3, random_state=1103)

# print the best parameters
lr_result.fit(Xtrain_selected_woe, y_train)
lr_result.best_params_

# %% [markdown]
# ## Model Evaluation

# %%
# Assgn the best parameters to logistic regression
lr_tuned = LogisticRegression(random_state=1103, class_weight='balanced', **lr_result.best_params_)
eval_model(lr_tuned, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, thresshold=0.42)

# %%
# assign best hyperparameter to lightgbm
lgbm_tuned = LGBMClassifier(random_state=1130, class_weight='balanced', **lgbm_result.best_params_)
eval_model(lgbm_tuned, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, thresshold=0.42)

# %%
def df_model_score(model, X_train, y_train, X_test, y_test, threshold):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_predict_proba = model.predict_proba(X_train)[:, 1]
    test_predic_proba = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    train_recall = recall_score(y_train, train_predict_proba > threshold)
    test_recall = recall_score(y_test, test_predic_proba > threshold)
    
    return train_auc, test_auc, train_recall, test_recall

# Assgn the best parameters to logistic regression
lr_tuned = LogisticRegression(random_state=1103, class_weight='balanced', **lr_result.best_params_)
# assign best hyperparameter to lightgbm
lgbm_tuned = LGBMClassifier(random_state=1130, class_weight='balanced', **lgbm_result.best_params_)

# Evaluate the models and store the scores
lgbm_scores = df_model_score(lgbm_tuned, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, threshold=0.42)
lr_scores = df_model_score(lr_tuned, X_train=Xtrain_selected_woe, y_train=y_train, X_test=Xtest_selected_woe, y_test=y_test, threshold=0.42)

# Create a DataFrame to store the scores
scores_df = pd.DataFrame(data=[lgbm_scores, lr_scores], 
                         columns=['Train AUC', 'Test AUC', 'Train Recall', 'Test Recall'], 
                         index=['LGBM', 'Logistic Regression'])

scores_df

# %%
# Define a custom palette
palette = {"Logistic Regression": "lightgray", "LGBM": "#6BFA00"}

# Melt the DataFrame to long format for seaborn
long_scores_df = scores_df.reset_index().melt(id_vars='index', var_name='Metrics', value_name='Scores')

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")  # Set the background color to white
barplot = sns.barplot(x='Metrics', y='Scores', hue='index', data=long_scores_df, palette=palette, edgecolor='black')

# Add annotations
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 10), 
                     textcoords = 'offset points')

plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.xticks(fontsize=12)
plt.legend(title='Model', title_fontsize='13', fontsize=12, bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(False)
plt.ylim(0, 0.98)
plt.show()

# %% [markdown]
# After Hyperparameter Tuning, we can see that the model has better performance than before, from the AUC score, and the Precision Score, Especially LightGBM model.

# %% [markdown]
# ## Confusion Matrix

# %%
pred_test = lgbm_tuned.predict_proba(Xtest_selected_woe)[:, 1]

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred_test > 0.42)

# Calculate the percentage of each category
percentage_matrix = cnf_matrix / cnf_matrix.sum()

# Define the labels
labels = np.array([['True Negative', 'False Positive'],
                   ['False Negative', 'True Positive']])

# Create labels with both count and percentage
new_labels = (np.asarray(["{0}\n{1} ({2:.2%})".format(label, value, percentage)
                          for label, value, percentage in zip(labels.flatten(), cnf_matrix.flatten(), percentage_matrix.flatten())])
             ).reshape(2,2)

# Plot confusion matrix using a heatmap
fig = plt.figure(figsize=(10,7))
fig.patch.set_facecolor('white')  # Change figure color

sns.heatmap(cnf_matrix, annot=new_labels, fmt='', cmap='viridis', annot_kws={"weight": "bold"})
# plt.title('Confusion Matrix LightGBM Model', fontsize=13, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Change label colors
plt.gca().xaxis.label.set_color('black')
plt.gca().yaxis.label.set_color('black')

plt.xticks(ticks=[0.5, 1.5], labels=['Good', 'Bad'], fontsize=16)
plt.yticks(ticks=[0.5, 1.5], labels=['Good', 'Bad'], fontsize=16)
plt.show()

# %%
# Define the initial variables
tp = 25378  # true positives
fn = 5465  # false negatives
total_clients = 1430153
total_clients_test = 357539
total_bad_clients_before = 123340
avg_loan = 600000  # average loan

# Calculate the default rate before the model
dr_before_model = total_bad_clients_before / total_clients
print(f'Total bad clients before model: {total_bad_clients_before}')
print(f'\nDefault rate before model: {dr_before_model}%')

# Calculate the default rate after the model
dr_after_model = ((0.5 * tp) + fn) / total_clients_test
print(f'Default rate after model: {round(dr_after_model * 100, 1)}%')

# Calculate the decrease in default rate
decreased_dr = dr_after_model - dr_before_model
print(f'Decreased default rate: {round(decreased_dr * 100, 1)}%')

# Calculate the total bad clients after the model
total_bad_clients_after = dr_after_model * total_clients
print(f'\nTotal bad clients after model: {round(total_bad_clients_after)}')

# Calculate the decrease in bad clients
decreased_bad_clients = total_bad_clients_after - total_bad_clients_before
print(f'Decreased bad clients: {round(decreased_bad_clients)}')

# Calculate the total good clients before the model
total_good_clients_before = total_clients - total_bad_clients_before

# Calculate the total revenue, bad debt and net revenue before the model
tr_before = total_good_clients_before * avg_loan
total_bad_debt_before = total_bad_clients_before * avg_loan
net_revenue_before = tr_before - total_bad_debt_before
print('\nTotal Revenue before model:', round(tr_before))
print(f'Total bad debt before model: {round(total_bad_debt_before)}')
print(f'Net revenue before model: {round(net_revenue_before)}')

# Calculate the total good clients after the model
total_good_clients_after = total_clients - total_bad_clients_after

# Calculate the total revenue, bad debt and net revenue after the model
total_bad_debt_after = total_bad_clients_after * avg_loan
tr_after = total_good_clients_after * avg_loan
net_revenue_after = tr_after - total_bad_debt_after
print('\nTotal Revenue after model:', round(tr_after))
print(f'Total bad debt after model: {round(total_bad_debt_after)}')
print(f'Net revenue after model: {round(net_revenue_after)}')

net_revenue_increase = net_revenue_after - net_revenue_before
print(f'\nNet revenue increase: {round(net_revenue_increase)}')

# %% [markdown]
# ## Feature Importances (LGBM)

# %%
# Get feature importances
importances = lgbm_tuned.feature_importances_

# Create a DataFrame for feature importances
feat_importances = pd.DataFrame({'Feature': Xtrain_selected_woe.columns, 'Importance': importances})

# Sort by importance
feat_importances = feat_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
feat_importances_sorted = feat_importances.sort_values(by='Importance', ascending=True)

# Define the number of top features you want to highlight
top_features = 5

# Create a color list, where the highest feature importance is highlighted in a different color
colors = ['skyblue' if (x < sorted(feat_importances_sorted['Importance'], reverse=True)[top_features-1]) else 'red' for x in feat_importances_sorted['Importance']]

plt.barh(feat_importances_sorted['Feature'], feat_importances_sorted['Importance'], color=colors)
sns.despine()
plt.grid(False)
plt.xlabel('Importance')
plt.title('Feature Importances', y=1.07)
plt.show()

# %%
# Create a TreeExplainer
explainer = shap.TreeExplainer(lgbm_tuned)

# Calculate SHAP values
shap_values = explainer.shap_values(Xtrain_selected_woe)

# %%
# Plot feature importance
shap.summary_plot(shap_values[1], Xtrain_selected_woe)

# %%
# Make predictions on the test data
probabilities = lgbm_tuned.predict_proba(Xtest_selected_woe)

# Select the probabilities of the positive class
positive_probabilities = probabilities[:, 1]

# Create a DataFrame with the IDs from the original dataset and the predicted probabilities
output = pd.DataFrame({
    'SK_ID_CURR': train_full.loc[Xtest_selected_woe.index, 'SK_ID_CURR'],
    'TARGET': positive_probabilities
})

# Save the output to a CSV file
output.to_csv('submission.csv', index=False)

# %%
# # Get the prediction probabilities for the positive class
# probabilities = lgbm_tuned.predict_proba(Xtrain_selected_woe)[:, 1]

# # Define the scaling factor and offset
# factor = 20 / np.log(2)
# offset = 600 - factor * np.log(20)

# # Scale the probabilities
# scores = probabilities * factor + offset

# # Calculate the ROC curve
# fpr, tpr, thresholds = roc_curve(y_train['TARGET'], scores)

# # Calculate the Youden's J statistic for each point on the ROC curve
# J = tpr - fpr

# # Find the optimal cutoff point
# optimal_idx = np.argmax(J)
# optimal_threshold = thresholds[optimal_idx]

# print(f'Optimal cutoff point: {optimal_threshold}')

# %% [markdown]
# ## Score Card model Experiment With Logistic Regression

# %%
# Get the coefficients and intercept from the logistic regression model
coefficients = lr_tuned.coef_[0]
intercept = lr_tuned.intercept_[0]

# Define the scaling factor and offset
pdo = 80
base_odds = 35
base_score = 1000
factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)

# Calculate the score for each feature
scorecard = pd.DataFrame(index=Xtrain_selected_woe.index)

for i in range(len(Xtrain_selected_woe.columns)):
    feature = Xtrain_selected_woe.columns[i]
    scorecard[feature] = (coefficients[i] * Xtrain_selected_woe[feature] + intercept / len(Xtrain_selected_woe.columns)) * factor + offset / len(Xtrain_selected_woe.columns)

# Calculate the total score for each row
scorecard['Total Score'] = scorecard.sum(axis=1)

# %%
y_scores = lr_tuned.predict_proba(Xtrain_selected_woe)[:, 1]

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]


min_score = scorecard['Total Score'].min()
max_score = scorecard['Total Score'].max()
optimal_score_cutoff = min_score + (optimal_threshold * (max_score - min_score))

# Classify the clients based on the optimal threshold
scorecard['Classification'] = ['good' if score >= optimal_score_cutoff else 'bad' for score in scorecard['Total Score']]

# %%
plt.figure(figsize=(10, 8))
sns.histplot(data=scorecard, x='Total Score', hue='Classification')
plt.axvline(x=optimal_score_cutoff, color='red', linestyle='--')
plt.title('Total Score Distribution')
plt.grid(False)
plt.show()

# %%
scorecard.Classification.value_counts()


