#!/usr/bin/env python
# coding: utf-8

# ## Insidoors XGBoost Classifier for PC Access, Building Access, and Proxy Logs
# This notebook details the second iteration of using XGBoost to identify suspicious employee activity. For ease of development, all classifiers are currently built and trained in this notebook. When integrating with the main application, a separate executable script should be created for each classifier.
# 
# #### Changelog
# 
# *Version: 2 (Current)*
# * Replaced data source with AWS RDS
# * Added option to normalize encodings as an alternative to sklearn's StandardScaler()
# * Implemented GridSearch for hyperparameter tuning
# * Enabled threshold tuning
# 
# *Version: 1*
# * Base classifier

# ### Load data

# In[ ]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# In[ ]:


# Create connection to RDS

conf = {
    'host': '<host>',
    'port': '<port>',
    'database': '<database>',
    'user': '<user>',
    'password': '<password>'
}

engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**conf))


# In[ ]:


# Load data from database into dataframes

with engine.connect() as con:
    employee_df = pd.read_sql('SELECT * FROM employees', con=con)
    pc_df = pd.read_sql('SELECT * FROM pc_access', con=con)
    building_df = pd.read_sql('SELECT * FROM building_access', con=con)
    proxy_df = pd.read_sql('SELECT * FROM proxy_log', con=con)

print('Employees:')
display(employee_df)

print('PC Access logs:')
display(pc_df)

print('Building Access logs:')
display(building_df)

print('Proxy logs:')
display(proxy_df)


# In[ ]:


# Add 'terminated' column to Employee dataframe

employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


# In[ ]:


# Inner join Employee dataframe with PC Access and Building Access dataframes

join_df_pc = employee_df[['id', 'location', 'terminated']].rename(
    columns={'location': 'user_location'}
)
join_df_building = employee_df[['id', 'terminated']]

pc_df = pc_df.join(join_df_pc.set_index('id'), on='user_id', how='inner')
building_df = building_df.join(join_df_building.set_index('id'), on='user_id', how='inner')


# In[ ]:


# Display distribution of suspect PC Access logs

print('Distribution of suspect PC Access logs:')
print(pc_df['suspect'].value_counts())
display(pc_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect PC Access logs

pc_cases = np.sort(pd.unique(pc_df['suspect']))

print('Suspect PC Access logs:')
display(pc_df[pc_df['suspect'].isin(pc_cases[1:])])


# In[ ]:


# Display distribution of suspect Building Access logs

print('Distribution of suspect Building Access logs:')
print(building_df['suspect'].value_counts())
display(building_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect Building Access logs

building_cases = np.sort(pd.unique(building_df['suspect']))

print('Suspect Building Access logs:')
display(building_df[building_df['suspect'].isin(building_cases[1:])])


# In[ ]:


# Display distribution of suspect Proxy logs

print('Distribution of suspect Proxy logs:')
print(proxy_df['suspect'].value_counts())
display(proxy_df['suspect'].value_counts().plot(kind='bar', rot=0))


# In[ ]:


# Display suspect Proxy logs

proxy_cases = np.sort(pd.unique(proxy_df['suspect']))

print('Suspect Proxy logs:')
display(proxy_df[proxy_df['suspect'].isin(proxy_cases[1:])])


# ### Specify settings and hyperparameters

# In[ ]:


# Whether to use sklearn's StandardScaler() to scale numeric data
useStandardScaler = False

# Parameter grid for GridSearch hyperparameter tuning
params = {
    'objective': ['binary:logitraw'],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
    'gamma': [0, 1, 2]
}

# Number of folds for GridSearch cross-validation
cv = 3


# ### Preprocess data

# In[ ]:


from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.creation import CyclicalFeatures


# In[ ]:


# Split 'access_date_time' column of PC Access dataframe into multiple component columns

datetime_format = '%Y-%m-%d %H:%M:%S'
pc_df['access_date_time'] = pd.to_datetime(pc_df['access_date_time'], format=datetime_format)

pc_df['access_year'] = pc_df['access_date_time'].map(lambda x: x.year)
pc_df['access_month'] = pc_df['access_date_time'].map(lambda x: x.month)
pc_df['access_day'] = pc_df['access_date_time'].map(lambda x: x.day)
pc_df['access_weekday'] = pc_df['access_date_time'].map(lambda x: x.weekday())
pc_df['access_hour'] = pc_df['access_date_time'].map(lambda x: x.hour)
pc_df['access_minute'] = pc_df['access_date_time'].map(lambda x: x.minute)
pc_df['access_second'] = pc_df['access_date_time'].map(lambda x: x.second)

display(pc_df)


# In[ ]:


# Split 'access_date_time' column of Building Access dataframe into multiple component columns

datetime_format = '%Y-%m-%d %H:%M:%S'
building_df['access_date_time'] = pd.to_datetime(building_df['access_date_time'], format=datetime_format)

building_df['access_year'] = building_df['access_date_time'].map(lambda x: x.year)
building_df['access_month'] = building_df['access_date_time'].map(lambda x: x.month)
building_df['access_day'] = building_df['access_date_time'].map(lambda x: x.day)
building_df['access_weekday'] = building_df['access_date_time'].map(lambda x: x.weekday())
building_df['access_hour'] = building_df['access_date_time'].map(lambda x: x.hour)
building_df['access_minute'] = building_df['access_date_time'].map(lambda x: x.minute)
building_df['access_second'] = building_df['access_date_time'].map(lambda x: x.second)

display(building_df)


# In[ ]:


# Split 'access_date_time' column of Proxy dataframe into multiple component columns

datetime_format = '%Y-%m-%d %H:%M:%S'
proxy_df['access_date_time'] = pd.to_datetime(proxy_df['access_date_time'], format=datetime_format)

proxy_df['access_year'] = proxy_df['access_date_time'].map(lambda x: x.year)
proxy_df['access_month'] = proxy_df['access_date_time'].map(lambda x: x.month)
proxy_df['access_day'] = proxy_df['access_date_time'].map(lambda x: x.day)
proxy_df['access_weekday'] = proxy_df['access_date_time'].map(lambda x: x.weekday())
proxy_df['access_hour'] = proxy_df['access_date_time'].map(lambda x: x.hour)
proxy_df['access_minute'] = proxy_df['access_date_time'].map(lambda x: x.minute)
proxy_df['access_second'] = proxy_df['access_date_time'].map(lambda x: x.second)

display(proxy_df)


# In[ ]:


# Define function to assign binary truth labels for anomaly detection

def label(input):
    if input == 0:
        return 0
    return 1


# In[ ]:


# Add binary 'label' column to PC Access dataframe

pc_df['label'] = pc_df['suspect'].map(label)


# In[ ]:


# Add binary 'label' column to Building Access dataframe

building_df['label'] = building_df['suspect'].map(label)


# In[ ]:


# Add binary 'label' column to Proxy dataframe

proxy_df['label'] = proxy_df['suspect'].map(label)


# In[ ]:


# Split PC Access dataframe into train and test
# 80% train, 20% test

pc_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'machine_lat', 'machine_long', 'suspect']
pc_X_train, pc_X_test, pc_y_train, pc_y_test = train_test_split(pc_df.drop(labels=pc_drop, axis=1),
                                                                pc_df['label'],
                                                                test_size=0.2,
                                                                random_state=480)

print('PC dataframe splits:')
print(pc_X_train.shape)
print(pc_X_test.shape)


# In[ ]:


# Split Building dataframe into train and test
# 80% train, 20% test

building_drop = ['id', 'user_id', 'access_date_time', 'office_lat', 'office_long', 'suspect']
building_X_train, building_X_test, building_y_train, building_y_test = train_test_split(building_df.drop(labels=building_drop, axis=1),
                                                                                        building_df['label'],
                                                                                        test_size=0.2,
                                                                                        random_state=480)

print('Building dataframe splits:')
print(building_X_train.shape)
print(building_X_test.shape)


# In[ ]:


# Split Proxy dataframe into train and test
# 80% train, 20% test

proxy_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'url', 'suspect']
proxy_X_train, proxy_X_test, proxy_y_train, proxy_y_test = train_test_split(proxy_df.drop(labels=proxy_drop, axis=1),
                                                                            proxy_df['label'],
                                                                            test_size=0.2,
                                                                            random_state=480)

print('Proxy dataframe splits:')
print(proxy_X_train.shape)
print(proxy_X_test.shape)


# In[ ]:


# Create feature encodings for PC Access logs

pc_numeric_features = ['access_year']
pc_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

pc_categorical_features = ['log_on_off', 'machine_location', 'working_hours',
                           'user_location', 'terminated']
pc_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
)

pc_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                        'access_hour', 'access_minute', 'access_second']
pc_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

if useStandardScaler:
    pc_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', pc_numeric_transformer, pc_numeric_features),
            ('categorical', pc_categorical_transformer, pc_categorical_features),
            ('cyclical', pc_cyclical_transformer, pc_cyclical_features)
            ],
        remainder='drop'
    )
else:
    pc_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', pc_categorical_transformer, pc_categorical_features),
            ('cyclical', pc_cyclical_transformer, pc_cyclical_features)
            ],
        remainder='passthrough'
    )

pc_preprocessor.fit(pc_X_train.drop(labels='label', axis=1))

print('PC dataframe columns after encoding:') 
print(pc_preprocessor.get_feature_names_out())

pc_train_encodings = pc_preprocessor.transform(pc_X_train.drop(labels='label', axis=1))

print('\nSample pc_train encoding:') 
print(pc_train_encodings[0])


# In[ ]:


# Create feature encodings for Building Access logs

building_numeric_features = ['attempts', 'access_year']
building_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

building_categorical_features = ['direction', 'status', 'office_location', 'terminated']
building_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
)

building_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                              'access_hour', 'access_minute', 'access_second']
building_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

if useStandardScaler:
    building_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', building_numeric_transformer, building_numeric_features),
            ('categorical', building_categorical_transformer, building_categorical_features),
            ('cyclical', building_cyclical_transformer, building_cyclical_features)
            ],
        remainder='drop'
    )
else:
    building_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', building_categorical_transformer, building_categorical_features),
            ('cyclical', building_cyclical_transformer, building_cyclical_features)
            ],
        remainder='passthrough'
    )

building_preprocessor.fit(building_X_train.drop(labels='label', axis=1))

print('Building dataframe columns after encoding:') 
print(building_preprocessor.get_feature_names_out())

building_train_encodings = building_preprocessor.transform(building_X_train.drop(labels='label', axis=1))

print('\nSample building_train encoding:') 
print(building_train_encodings[0])


# In[ ]:


# Create feature encodings for Proxy logs

proxy_numeric_features = ['bytes_in', 'bytes_out', 'access_year']
proxy_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

proxy_categorical_features = ['category']
proxy_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
)

proxy_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                           'access_hour', 'access_minute', 'access_second']
proxy_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

if useStandardScaler:
    proxy_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', proxy_numeric_transformer, proxy_numeric_features),
            ('categorical', proxy_categorical_transformer, proxy_categorical_features),
            ('cyclical', proxy_cyclical_transformer, proxy_cyclical_features)
            ],
        remainder='drop'
    )
else:
    proxy_preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', proxy_categorical_transformer, proxy_categorical_features),
            ('cyclical', proxy_cyclical_transformer, proxy_cyclical_features)
            ],
        remainder='passthrough'
    )

proxy_preprocessor.fit(proxy_X_train.drop(labels='label', axis=1))

print('Proxy dataframe columns after encoding:') 
print(proxy_preprocessor.get_feature_names_out())

proxy_train_encodings = proxy_preprocessor.transform(proxy_X_train.drop(labels='label', axis=1))

print('\nSample proxy_train encoding:') 
print(proxy_train_encodings[0])


# ### Train anomaly detection classifier with XGBoost

# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# In[ ]:


# Define function to normalize encodings

def normalize(x):
    m_ = np.mean(x, axis=1, keepdims=True)
    x = x - m_
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    return x


# In[ ]:


# Normalize encodings if StandardScaler is not used

pc_test_encodings = pc_preprocessor.transform(pc_X_test.drop(labels='label', axis=1))
building_test_encodings = building_preprocessor.transform(building_X_test.drop(labels='label', axis=1))
proxy_test_encodings = proxy_preprocessor.transform(proxy_X_test.drop(labels='label', axis=1))

if not useStandardScaler:
    pc_train_encodings = normalize(pc_train_encodings)
    pc_test_encodings = normalize(pc_test_encodings)
    
    building_train_encodings = normalize(building_train_encodings)
    building_test_encodings = normalize(building_test_encodings)
    
    proxy_train_encodings = normalize(proxy_train_encodings)
    proxy_test_encodings = normalize(proxy_test_encodings)


# In[ ]:


# Compute positive weight for PC Access classifier

pc_counts = pc_X_train['label'].value_counts()
pc_weight = pc_counts[0] / pc_counts[1]
print(pc_weight)


# In[ ]:


# Initialise and fit XGBoost for PC Access logs
# Scoring uses roc_auc

pc_clf = GridSearchCV(XGBClassifier(scale_pos_weight=pc_weight, eval_metric='auc'),
                      param_grid=params,
                      scoring='roc_auc',
                      cv=cv,
                      verbose=3)

pc_clf.fit(pc_train_encodings, pc_y_train)


# In[ ]:


# Run predictions for PC Access logs

print(pc_clf.best_params_)

pc_test_scores = pc_clf.predict_proba(pc_test_encodings)[:,1]
pc_test_preds = pc_clf.predict(pc_test_encodings)


# In[ ]:


# Compute positive weight for Building Access classifier

building_counts = building_X_train['label'].value_counts()
building_weight = building_counts[0] / building_counts[1]
print(building_weight)


# In[ ]:


# Initialise and fit XGBoost for Building Access logs
# Scoring uses roc_auc

building_clf = GridSearchCV(XGBClassifier(scale_pos_weight=building_weight, eval_metric='auc'),
                            param_grid=params,
                            scoring='roc_auc',
                            cv=cv,
                            verbose=3)

building_clf.fit(building_train_encodings, building_y_train)


# In[ ]:


# Run predictions for Building Access logs

print(building_clf.best_params_)

building_test_scores = building_clf.predict_proba(building_test_encodings)[:,1]
building_test_preds = building_clf.predict(building_test_encodings)


# In[ ]:


# Compute positive weight for Proxy classifier

proxy_counts = proxy_X_train['label'].value_counts()
proxy_weight = proxy_counts[0] / proxy_counts[1]
print(proxy_weight)


# In[ ]:


# Initialise and fit XGBoost for Proxy logs
# Scoring uses roc_auc

proxy_clf = GridSearchCV(XGBClassifier(scale_pos_weight=proxy_weight, eval_metric='auc'),
                         param_grid=params,
                         scoring='roc_auc',
                         cv=cv,
                         verbose=3)

proxy_clf.fit(proxy_train_encodings, proxy_y_train)


# In[ ]:


# Run predictions for Proxy logs

print(proxy_clf.best_params_)

proxy_test_scores = proxy_clf.predict_proba(proxy_test_encodings)[:,1]
proxy_test_preds = proxy_clf.predict(proxy_test_encodings)


# ### Evaluate classifier

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[ ]:


cm_labels = [0, 1]
cm_display_labels = ['normal', 'anomaly']


# In[ ]:


# Compute metrics for PC classifier

pc_test_accuracy = accuracy_score(pc_y_test, pc_test_preds)
pc_test_precision = precision_score(pc_y_test, pc_test_preds)
pc_test_recall = recall_score(pc_y_test, pc_test_preds)
pc_test_f1 = f1_score(pc_y_test, pc_test_preds)
pc_test_f1_weighted = f1_score(pc_y_test, pc_test_preds, average='weighted')
pc_test_roc_auc = roc_auc_score(pc_y_test, pc_test_scores)
pc_test_confusion = confusion_matrix(pc_y_test, pc_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for PC classifier

print('Results for XGBoost on pc_test:')
print('Accuracy:', pc_test_accuracy)
print('Precision:', pc_test_precision)
print('Recall:', pc_test_recall)
print('F1:', pc_test_f1)
print('F1 (Weighted):', pc_test_f1_weighted)
print('ROC AUC:', pc_test_roc_auc)

pc_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=pc_test_confusion,
                                                display_labels=cm_display_labels)
pc_test_confusion_disp.plot(values_format='')
plt.show()


# In[ ]:


# Compute metrics for Building classifier

building_test_accuracy = accuracy_score(building_y_test, building_test_preds)
building_test_precision = precision_score(building_y_test, building_test_preds)
building_test_recall = recall_score(building_y_test, building_test_preds)
building_test_f1 = f1_score(building_y_test, building_test_preds)
building_test_f1_weighted = f1_score(building_y_test, building_test_preds, average='weighted')
building_test_roc_auc = roc_auc_score(building_y_test, building_test_scores)
building_test_confusion = confusion_matrix(building_y_test, building_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for Building classifier

print('Results for XGBoost on building_test:')
print('Accuracy:', building_test_accuracy)
print('Precision:', building_test_precision)
print('Recall:', building_test_recall)
print('F1:', building_test_f1)
print('F1 (Weighted):', building_test_f1_weighted)
print('ROC AUC:', building_test_roc_auc)

building_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=building_test_confusion,
                                                      display_labels=cm_display_labels)
building_test_confusion_disp.plot(values_format='')
plt.show()


# In[ ]:


# Compute metrics for Proxy classifier

proxy_test_accuracy = accuracy_score(proxy_y_test, proxy_test_preds)
proxy_test_precision = precision_score(proxy_y_test, proxy_test_preds)
proxy_test_recall = recall_score(proxy_y_test, proxy_test_preds)
proxy_test_f1 = f1_score(proxy_y_test, proxy_test_preds)
proxy_test_f1_weighted = f1_score(proxy_y_test, proxy_test_preds, average='weighted')
proxy_test_roc_auc = roc_auc_score(proxy_y_test, proxy_test_scores)
proxy_test_confusion = confusion_matrix(proxy_y_test, proxy_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for Proxy classifier

print('Results for XGBoost on proxy_test:')
print('Accuracy:', proxy_test_accuracy)
print('Precision:', proxy_test_precision)
print('Recall:', proxy_test_recall)
print('F1:', proxy_test_f1)
print('F1 (Weighted):', proxy_test_f1_weighted)
print('ROC AUC:', proxy_test_roc_auc)

proxy_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=proxy_test_confusion,
                                                   display_labels=cm_display_labels)
proxy_test_confusion_disp.plot(values_format='')
plt.show()


# ### Save classifier

# In[ ]:


from joblib import dump, load


# In[ ]:


# Save classifiers

path = 'models/xgb/'

if useStandardScaler:
    PC_CLF = path + 'pc_xgb_scale.joblib'
    BUILDING_CLF = path + 'building_xgb_scale.joblib'
    PROXY_CLF = path + 'proxy_xgb_scale.joblib'
else:
    PC_CLF = path + 'pc_xgb_norm.joblib'
    BUILDING_CLF = path + 'building_xgb_norm.joblib'
    PROXY_CLF = path + 'proxy_xgb_norm.joblib'

dump(pc_clf, PC_CLF)
dump(building_clf, BUILDING_CLF)
dump(proxy_clf, PROXY_CLF)


# ### Test classifier predictions on unseen cases

# In[ ]:


from collections import Counter


# In[ ]:


# Load classifiers

pc_clf_loaded = load(PC_CLF)
building_clf_loaded = load(BUILDING_CLF)
proxy_clf_loaded = load(PROXY_CLF)


# In[ ]:


# Create new data for PC Access logs
# New case: unseen machine location

countries = ['Russia', 'China', 'India', 'Brazil']

pc_unseen = pc_X_test[pc_X_test['label'] == 0].copy()

np.random.seed(480)
pc_unseen['machine_location'] = np.random.choice(countries, pc_unseen.shape[0])

display(pc_unseen)


# In[ ]:


# Create new data for Building Access logs
# New case: large number of access attempts

building_unseen = building_X_test[building_X_test['label'] == 0].copy()

np.random.seed(480)
building_unseen['attempts'] = np.random.randint(6, 20, building_unseen.shape[0])

display(building_unseen)


# In[ ]:


# Create new data for Proxy logs
# New case: data upload/download from malicious urls

url_types = ['malware', 'phishing']

malicious_urls_df = pd.read_csv('malicious_urls.csv', sep=',', header=0)
malicious_urls_df = malicious_urls_df[malicious_urls_df['type'].isin(url_types)]

display(malicious_urls_df)

urls = malicious_urls_df['url']

proxy_unseen = proxy_X_test[proxy_X_test['label'] == 0].copy()

np.random.seed(480)
# proxy_unseen['url'] = np.random.choice(urls, proxy_unseen.shape[0])
proxy_unseen['category'] = 'Malware, Phishing'

display(proxy_unseen)


# In[ ]:


# Compute accuracy of PC classifier on unseen case

pc_unseen_encodings = pc_preprocessor.transform(pc_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    pc_unseen_encodings = normalize(pc_unseen_encodings)

pc_unseen_scores = pc_clf_loaded.predict_proba(pc_unseen_encodings)[:,1]
pc_unseen_preds = pc_clf_loaded.predict(pc_unseen_encodings)

pc_results_counter = Counter(pc_unseen_preds)

print('PC classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[1], len(pc_unseen_preds), pc_results_counter[1]/len(pc_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Building classifier on unseen case

building_unseen_encodings = building_preprocessor.transform(building_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    building_unseen_encodings = normalize(building_unseen_encodings)

building_unseen_scores = building_clf_loaded.predict_proba(building_unseen_encodings)[:,1]
building_unseen_preds = building_clf_loaded.predict(building_unseen_encodings)

building_results_counter = Counter(building_unseen_preds)

print('Building classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[1], len(building_unseen_preds), building_results_counter[1]/len(building_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Proxy classifier on unseen case

proxy_unseen_encodings = proxy_preprocessor.transform(proxy_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    proxy_unseen_encodings = normalize(proxy_unseen_encodings)

proxy_unseen_scores = proxy_clf_loaded.predict_proba(proxy_unseen_encodings)[:,1]
proxy_unseen_preds = proxy_clf_loaded.predict(proxy_unseen_encodings)

proxy_results_counter = Counter(proxy_unseen_preds)

print('Proxy classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[1], len(proxy_unseen_preds), proxy_results_counter[1]/len(proxy_unseen_preds)*100))


# ### Threshold analysis

# In[ ]:


# Display score distributions for PC classifier

print('PC seen:')
plt.hist(pc_test_scores, bins=100)
plt.show()

print('PC unseen:')
plt.hist(pc_unseen_scores, bins=100)
plt.show()


# In[ ]:


# Manually set custom threshold for PC classifier

# Default
print('PC confusion matrix for test data (Default threshold):')
pc_test_confusion_disp.plot(values_format='')
plt.show()

print('PC classifier accuracy for unseen data (Default threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[1], len(pc_unseen_preds),
       pc_results_counter[1] / len(pc_unseen_preds) * 100))

# Custom
pc_threshold = -4.75

pc_test_preds_manual = np.where(pc_test_scores < pc_threshold, 0, 1)
pc_test_confusion_manual = confusion_matrix(pc_y_test, pc_test_preds_manual, labels=cm_labels)
pc_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=pc_test_confusion_manual,
                                                       display_labels=cm_display_labels)

print('\nPC confusion matrix for test data (Custom threshold):')
pc_test_confusion_disp_manual.plot(values_format='')
plt.show()

pc_unseen_preds_manual = np.where(pc_unseen_scores < pc_threshold, 0, 1)
pc_results_counter_manual = Counter(pc_unseen_preds_manual)

print('PC classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter_manual[1], len(pc_unseen_preds_manual),
       pc_results_counter_manual[1] / len(pc_unseen_preds_manual) * 100))


# In[ ]:


# Display score distributions for Building classifier

print('Building seen:')
plt.hist(building_test_scores, bins=100)
plt.show()

print('Building unseen:')
plt.hist(building_unseen_scores, bins=100)
plt.show()


# In[ ]:


# Manually set custom threshold for Building classifier

# Default
print('Building confusion matrix for test data (Default threshold):')
building_test_confusion_disp.plot(values_format='')
plt.show()

print('Building classifier accuracy for unseen data (Default threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[1], len(building_unseen_preds),
       building_results_counter[1] / len(building_unseen_preds) * 100))

# Custom
building_threshold = 0.25

building_test_preds_manual = np.where(building_test_scores < building_threshold, 0, 1)
building_test_confusion_manual = confusion_matrix(building_y_test, building_test_preds_manual, labels=cm_labels)
building_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=building_test_confusion_manual,
                                                             display_labels=cm_display_labels)

print('\nBuilding confusion matrix for test data (Custom threshold):')
building_test_confusion_disp_manual.plot(values_format='')
plt.show()

building_unseen_preds_manual = np.where(building_unseen_scores < building_threshold, 0, 1)
building_results_counter_manual = Counter(building_unseen_preds_manual)

print('Building classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter_manual[1], len(building_unseen_preds_manual),
       building_results_counter_manual[1] / len(building_unseen_preds_manual) * 100))


# In[ ]:


# Display score distributions for Proxy classifier

print('Proxy seen:')
plt.hist(proxy_test_scores, bins=100)
plt.show()

print('Proxy unseen:')
plt.hist(proxy_unseen_scores, bins=100)
plt.show()


# In[ ]:


# Manually set custom threshold for Proxy classifier

# Default
print('Proxy confusion matrix for test data (Default threshold):')
proxy_test_confusion_disp.plot(values_format='')
plt.show()

print('Proxy classifier accuracy for unseen data (Default threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[1], len(proxy_unseen_preds),
       proxy_results_counter[1] / len(proxy_unseen_preds) * 100))

# Custom
proxy_threshold = -4

proxy_test_preds_manual = np.where(proxy_test_scores < proxy_threshold, 0, 1)
proxy_test_confusion_manual = confusion_matrix(proxy_y_test, proxy_test_preds_manual, labels=cm_labels)
proxy_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=proxy_test_confusion_manual,
                                                          display_labels=cm_display_labels)

print('\nProxy confusion matrix for test data (Custom threshold):')
proxy_test_confusion_disp_manual.plot(values_format='')
plt.show()

proxy_unseen_preds_manual = np.where(proxy_unseen_scores < proxy_threshold, 0, 1)
proxy_results_counter_manual = Counter(proxy_unseen_preds_manual)

print('Proxy classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter_manual[1], len(proxy_unseen_preds_manual),
       proxy_results_counter_manual[1] / len(proxy_unseen_preds_manual) * 100))

