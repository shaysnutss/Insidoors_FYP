#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Required python modules:
# - feature-engine
# - xgboost


# In[ ]:


# !pip3 install feature-engine
# !pip3 install xgboost


# In[ ]:


# FUTURE ITERATIONS:
# 1. Use gridsearch for hyperparameter tuning


# ## Insidoors XGBoost Classifier for PC Access, Building Access, and Proxy Logs
# This notebook details the first iteration of using XGBoost to identify suspicious employee activity. For ease of development, all classifiers are currently built and trained in this notebook. In the future, a separate notebook will be created for each classifier.
# 
# #### Changelog
# 
# *Version: 1 (Current)*
# * Base classifier

# ### Load data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


employee_df = pd.read_csv('employee_data_5k.csv', sep=';', header=0)
pc_df = pd.read_csv('pc_data_5k_modified.csv', sep=';', header=0)
building_df = pd.read_csv('building_data_5k_modified.csv', sep=';', header=0)
proxy_df = pd.read_csv('proxy_data_5k_modified.csv', sep=';', header=0)

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


# Preparation for testing unseen cases
# Add 'attempts' column to Building Access dataframe

np.random.seed(480)
building_df['attempts'] = np.random.randint(1, 6, building_df.shape[0])


# In[ ]:


# Remove cases 2 and 5 from all logs

# to_exclude = [2, 5]

# pc_df = pc_df[~pc_df['suspect'].isin(to_exclude)]
# building_df = building_df[~building_df['suspect'].isin(to_exclude)]
# proxy_df = proxy_df[~proxy_df['suspect'].isin(to_exclude)]


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
    if input != 0:
        return 1
    return 0


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

pc_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'label']
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

building_drop = ['id', 'user_id', 'access_date_time', 'label']
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

proxy_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'url', 'label']
proxy_X_train, proxy_X_test, proxy_y_train, proxy_y_test = train_test_split(proxy_df.drop(labels=proxy_drop, axis=1),
                                                                            proxy_df['label'],
                                                                            test_size=0.2,
                                                                            random_state=480)

print('Proxy dataframe splits:')
print(proxy_X_train.shape)
print(proxy_X_test.shape)


# In[ ]:


# Create feature encoding pipeline for PC Access logs

pc_numeric_features = ['access_year']
pc_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

pc_categorical_features = ['log_on_off', 'machine_location', 'user_location', 'terminated']
pc_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]
)

pc_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                        'access_hour', 'access_minute', 'access_second']
pc_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

pc_preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', pc_numeric_transformer, pc_numeric_features),
        ('categorical', pc_categorical_transformer, pc_categorical_features),
        ('cyclical', pc_cyclical_transformer, pc_cyclical_features)
    ],
    remainder='drop'
)

pc_preprocessor.fit(pc_X_train)

print('PC dataframe columns after encoding:') 
print(pc_preprocessor.get_feature_names_out())

pc_train_encodings = pc_preprocessor.transform(pc_X_train)

print('\nSample pc_train encoding:') 
print(pc_train_encodings[0])


# In[ ]:


# Create feature encoding pipeline for Building Access logs

building_numeric_features = ['attempts', 'access_year']
building_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

building_categorical_features = ['direction', 'status', 'office_location', 'terminated']
building_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]
)

building_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                              'access_hour', 'access_minute', 'access_second']
building_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

building_preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', building_numeric_transformer, building_numeric_features),
        ('categorical', building_categorical_transformer, building_categorical_features),
        ('cyclical', building_cyclical_transformer, building_cyclical_features)
    ],
    remainder='drop'
)

building_preprocessor.fit(building_X_train)

print('Building dataframe columns after encoding:') 
print(building_preprocessor.get_feature_names_out())

building_train_encodings = building_preprocessor.transform(building_X_train)

print('\nSample building_train encoding:') 
print(building_train_encodings[0])


# In[ ]:


# Create feature encoding pipeline for Proxy logs

proxy_numeric_features = ['bytes_in', 'bytes_out', 'access_year']
proxy_numeric_transformer = Pipeline(
    steps=[('scaler', StandardScaler())]
)

proxy_categorical_features = ['category']
proxy_categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))]
)

proxy_cyclical_features = ['access_month', 'access_day', 'access_weekday',
                           'access_hour', 'access_minute', 'access_second']
proxy_cyclical_transformer = Pipeline(
    steps=[('encoder', CyclicalFeatures(drop_original=True))]
)

proxy_preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', proxy_numeric_transformer, proxy_numeric_features),
        ('categorical', proxy_categorical_transformer, proxy_categorical_features),
        ('cyclical', proxy_cyclical_transformer, proxy_cyclical_features)
    ],
    remainder='drop'
)

proxy_preprocessor.fit(proxy_X_train)

print('Proxy dataframe columns after encoding:') 
print(proxy_preprocessor.get_feature_names_out())

proxy_train_encodings = proxy_preprocessor.transform(proxy_X_train)

print('\nSample proxy_train encoding:') 
print(proxy_train_encodings[0])


# ### Train anomaly detection classifier with XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


# Initialise and fit XGBoost classifier for PC Access logs

pc_clf = Pipeline(
    steps=[
        ('preprocessor', pc_preprocessor),
        ('classifier', XGBClassifier(objective='binary:logistic'))
    ]
)

pc_clf.fit(pc_X_train, pc_y_train)


# In[ ]:


# Run predictions for PC Access logs

# pc_train_preds = pc_clf.predict(pc_X_train)
pc_test_preds = pc_clf.predict(pc_X_test)


# In[ ]:


# Initialise and fit XGBoost classifier for Building Access logs

building_clf = Pipeline(
    steps=[
        ('preprocessor', building_preprocessor),
        ('classifier', XGBClassifier(objective='binary:logistic'))
    ]
)

building_clf.fit(building_X_train, building_y_train)


# In[ ]:


# Run predictions for Building Access logs

# building_train_preds = building_clf.predict(building_X_train)
building_test_preds = building_clf.predict(building_X_test)


# In[ ]:


# Initialise and fit XGBoost classifier for Proxy Access logs

proxy_clf = Pipeline(
    steps=[
        ('preprocessor', proxy_preprocessor),
        ('classifier', XGBClassifier(objective='binary:logistic'))
    ]
)

proxy_clf.fit(proxy_X_train, proxy_y_train)


# In[ ]:


# Run predictions for Proxy logs

# proxy_train_preds = proxy_clf.predict(proxy_X_train)
proxy_test_preds = proxy_clf.predict(proxy_X_test)


# ### Evaluate classifier

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay


# In[ ]:


cm_labels = [0, 1]
cm_display_labels = ['normal', 'anomaly']


# In[ ]:


# Compute metrics for PC classifier

# pc_train_accuracy = accuracy_score(pc_y_train, pc_train_preds)
# pc_train_precision = precision_score(pc_y_train, pc_train_preds)
# pc_train_recall = recall_score(pc_y_train, pc_train_preds)
# pc_train_f1 = f1_score(pc_y_train, pc_train_preds)
# pc_train_f1_weighted = f1_score(pc_y_train, pc_train_preds, average='weighted')
# pc_train_confusion = confusion_matrix(pc_y_train, pc_train_preds, labels=cm_labels)

pc_test_accuracy = accuracy_score(pc_y_test, pc_test_preds)
pc_test_precision = precision_score(pc_y_test, pc_test_preds)
pc_test_recall = recall_score(pc_y_test, pc_test_preds)
pc_test_f1 = f1_score(pc_y_test, pc_test_preds)
pc_test_f1_weighted = f1_score(pc_y_test, pc_test_preds, average='weighted')
pc_test_confusion = confusion_matrix(pc_y_test, pc_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for pc_train

# print('Results for XGBoost on pc_train:')
# print('Accuracy:', pc_train_accuracy)
# print('Precision:', pc_train_precision)
# print('Recall:', pc_train_recall)
# print('F1:', pc_train_f1)
# print('F1 (Weighted):', pc_train_f1_weighted)

# print('\nConfusion Matrix:')
# pc_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=pc_train_confusion,
#                                                  display_labels=cm_display_labels)
# pc_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for pc_test

print('Results for XGBoost on pc_test:')
print('Accuracy:', pc_test_accuracy)
print('Precision:', pc_test_precision)
print('Recall:', pc_test_recall)
print('F1:', pc_test_f1)
print('F1 (Weighted):', pc_test_f1_weighted)
print('Confusion Matrix: \n', pc_test_confusion)

pc_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=pc_test_confusion,
                                                display_labels=cm_display_labels)
pc_test_confusion_disp.plot()
plt.show()


# In[ ]:


# Compute metrics for Building classifier

# building_train_accuracy = accuracy_score(building_y_train, building_train_preds)
# building_train_precision = precision_score(building_y_train, building_train_preds)
# building_train_recall = recall_score(building_y_train, building_train_preds)
# building_train_f1 = f1_score(building_y_train, building_train_preds)
# building_train_f1_weighted = f1_score(building_y_train, building_train_preds, average='weighted')
# building_train_confusion = confusion_matrix(building_y_train, building_train_preds, labels=cm_labels)

building_test_accuracy = accuracy_score(building_y_test, building_test_preds)
building_test_precision = precision_score(building_y_test, building_test_preds)
building_test_recall = recall_score(building_y_test, building_test_preds)
building_test_f1 = f1_score(building_y_test, building_test_preds)
building_test_f1_weighted = f1_score(building_y_test, building_test_preds, average='weighted')
building_test_confusion = confusion_matrix(building_y_test, building_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for building_train

# print('Results for XGBoost on building_train:')
# print('Accuracy:', building_train_accuracy)
# print('Precision:', building_train_precision)
# print('Recall:', building_train_recall)
# print('F1:', building_train_f1)
# print('F1 (Weighted):', building_train_f1_weighted)
# print('Confusion Matrix: \n', building_train_confusion)

# building_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=building_train_confusion,
#                                                        display_labels=cm_display_labels)
# building_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for building_test

print('Results for XGBoost on building_test:')
print('Accuracy:', building_test_accuracy)
print('Precision:', building_test_precision)
print('Recall:', building_test_recall)
print('F1:', building_test_f1)
print('F1 (Weighted):', building_test_f1_weighted)
print('Confusion Matrix: \n', building_test_confusion)

building_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=building_test_confusion,
                                                      display_labels=cm_display_labels)
building_test_confusion_disp.plot()
plt.show()


# In[ ]:


# Compute metrics for Proxy classifier

# proxy_train_accuracy = accuracy_score(proxy_y_train, proxy_train_preds)
# proxy_train_precision = precision_score(proxy_y_train, proxy_train_preds)
# proxy_train_recall = recall_score(proxy_y_train, proxy_train_preds)
# proxy_train_f1 = f1_score(proxy_y_train, proxy_train_preds)
# proxy_train_f1_weighted = f1_score(proxy_y_train, proxy_train_preds, average='weighted')
# proxy_train_confusion = confusion_matrix(proxy_y_train, proxy_train_preds, labels=cm_labels)

proxy_test_accuracy = accuracy_score(proxy_y_test, proxy_test_preds)
proxy_test_precision = precision_score(proxy_y_test, proxy_test_preds)
proxy_test_recall = recall_score(proxy_y_test, proxy_test_preds)
proxy_test_f1 = f1_score(proxy_y_test, proxy_test_preds)
proxy_test_f1_weighted = f1_score(proxy_y_test, proxy_test_preds, average='weighted')
proxy_test_confusion = confusion_matrix(proxy_y_test, proxy_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for proxy_train

# print('Results for XGBoost on proxy_train:')
# print('Accuracy:', proxy_train_accuracy)
# print('Precision:', proxy_train_precision)
# print('Recall:', proxy_train_recall)
# print('F1:', proxy_train_f1)
# print('F1 (Weighted):', proxy_train_f1_weighted)
# print('Confusion Matrix: \n', proxy_train_confusion)

# proxy_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=proxy_train_confusion,
#                                                     display_labels=cm_display_labels)
# proxy_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for proxy_test

print('Results for XGBoost on proxy_test:')
print('Accuracy:', proxy_test_accuracy)
print('Precision:', proxy_test_precision)
print('Recall:', proxy_test_recall)
print('F1:', proxy_test_f1)
print('F1 (Weighted):', proxy_test_f1_weighted)
print('Confusion Matrix: \n', proxy_test_confusion)

proxy_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=proxy_test_confusion,
                                                   display_labels=cm_display_labels)
proxy_test_confusion_disp.plot()
plt.show()


# ### Test classifier predictions on unseen cases

# In[ ]:


from collections import Counter


# In[ ]:


# Create new data for PC Access logs
# New case: unseen machine location

countries = ['Russia', 'China', 'India', 'Brazil']

pc_unseen = pc_X_test[pc_X_test['suspect'] == 0].copy()

np.random.seed(480)
pc_unseen['machine_location'] = np.random.choice(countries, pc_unseen.shape[0])

display(pc_unseen)


# In[ ]:


# Create new data for Building Access logs
# New case: large number of access attempts

building_unseen = building_X_test[building_X_test['suspect'] == 0].copy()

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

proxy_unseen = proxy_X_test[proxy_X_test['suspect'] == 0].copy()

np.random.seed(480)
# proxy_unseen['url'] = np.random.choice(urls, proxy_unseen.shape[0])
proxy_unseen['category'] = 'Malware, Phishing'

display(proxy_unseen)


# In[ ]:


# Compute accuracy of PC classifier on unseen case

pc_unseen_preds = pc_clf.predict(pc_unseen)

pc_results_counter = Counter(pc_unseen_preds)

print('PC classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[1], len(pc_unseen_preds), pc_results_counter[1]/len(pc_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Building classifier on unseen case

building_unseen_preds = building_clf.predict(building_unseen)

building_results_counter = Counter(building_unseen_preds)

print('Building classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[1], len(building_unseen_preds), building_results_counter[1]/len(building_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Proxy classifier on unseen case

proxy_unseen_preds = proxy_clf.predict(proxy_unseen)

proxy_results_counter = Counter(proxy_unseen_preds)

print('Proxy classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[1], len(proxy_unseen_preds), proxy_results_counter[1]/len(proxy_unseen_preds)*100))

