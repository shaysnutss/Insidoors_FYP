#!/usr/bin/env python
# coding: utf-8

# ## Insidoors Multinomial Naive Bayes Classifier for PC Access, Building Access, and Proxy Logs
# This notebook details the second iteration of using Multinomial Naive Bayes to identify suspicious employee activity. For ease of development, all classifiers are currently built and trained in this notebook. When integrating with the main application, a separate executable script should be created for each classifier.
# 
# #### Changelog
# 
# *Version: 2 (Current)*
# * Replaced data source with AWS RDS
# * Added option to normalize encodings as an alternative to sklearn's StandardScaler()
# * Implemented GridSearch for hyperparameter tuning
# * Enabled threshold tuning
# 
# *Version: 1 (Current)*
# * Base

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


# Whether to use CountVectorizer() or TfidfVectorizer() to vectorize input
# Acceptable values: {count, tfidf}
vectorizer = 'tfidf'

# Parameter grid for GridSearch hyperparameter tuning
params = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5]
}

# Number of folds for GridSearch cross-validation
cv = 3


# ### Preprocess data

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[ ]:


# Concatenate input columns of PC Access logs into a single string and create label column

pc_df = pc_df.astype(str)
pc_df['suspect'] = pc_df['suspect'].astype(int) # Keep 'suspect' column as int
pc_df['input'] = pc_df[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                        'machine_location', 'working_hours', 'user_location', 'terminated']].agg(', '.join, axis=1)
pc_df['label'] = pc_df['suspect']

display(pc_df)


# In[ ]:


# Concatenate input columns of Building Access logs into a single string and create label column

building_df = building_df.astype(str)
building_df['suspect'] = building_df['suspect'].astype(int) # Keep 'suspect' column as int
building_df['input'] = building_df[['user_id', 'access_date_time', 'direction', 'status',
                                    'office_location', 'attempts', 'terminated']].agg(', '.join, axis=1)
building_df['label'] = building_df['suspect']

display(building_df)


# In[ ]:


# Concatenate input columns of Proxy logs into a single string and create label column

proxy_df = proxy_df.astype(str)
proxy_df['suspect'] = proxy_df['suspect'].astype(int) # Keep 'suspect' column as int
# proxy_df['input'] = proxy_df[['user_id', 'access_date_time', 'machine_name',
#                               'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)
proxy_df['input'] = proxy_df[['user_id', 'access_date_time', 'machine_name',
                              'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)
proxy_df['label'] = proxy_df['suspect']

display(proxy_df)


# In[ ]:


# Define function to assign binary truth labels for anomaly detection

def label(input):
    if input == 0:
        return 1
    return -1


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

pc_train, pc_test = np.split(pc_df.sample(frac=1, random_state=480),
                             [int(0.8 * len(pc_df))])

print('PC dataframe splits:')
print(pc_train.shape)
print(pc_test.shape)


# In[ ]:


# Split Building dataframe into train and test
# 80% train, 20% test

building_train, building_test = np.split(building_df.sample(frac=1, random_state=480),
                                        [int(0.8 * len(building_df))])

print('Building dataframe splits:')
print(building_train.shape)
print(building_test.shape)


# In[ ]:


# Split Proxy dataframe into train and test
# 80% train, 20% test

proxy_train, proxy_test = np.split(proxy_df.sample(frac=1, random_state=480),
                                  [int(0.8 * len(proxy_df))])

print('Proxy dataframe splits:')
print(proxy_train.shape)
print(proxy_test.shape)


# In[ ]:


# Initialise tf-idf vectorizers

if vectorizer == 'count':
    pc_vectorizer = CountVectorizer()
    building_vectorizer = CountVectorizer()
    proxy_vectorizer = CountVectorizer()
elif vectorizer == 'tfidf':
    pc_vectorizer = TfidfVectorizer()
    building_vectorizer = TfidfVectorizer()
    proxy_vectorizer = TfidfVectorizer()


# In[ ]:


# Vectorise input text

pc_train_vectors = pc_vectorizer.fit_transform(pc_train['input'])
pc_test_vectors = pc_vectorizer.transform(pc_test['input'])

building_train_vectors = building_vectorizer.fit_transform(building_train['input'])
building_test_vectors = building_vectorizer.transform(building_test['input'])

proxy_train_vectors = proxy_vectorizer.fit_transform(proxy_train['input'])
proxy_test_vectors = proxy_vectorizer.transform(proxy_test['input'])


# ### Train anomaly detection classifier with Multinomial NB

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


# Initialise and fit Multinomial NB for PC Access logs
# Scoring uses roc_auc

pc_clf = GridSearchCV(MultinomialNB(),
                      param_grid=params,
                      scoring='roc_auc',
                      cv=cv,
                      verbose=3)

pc_clf.fit(pc_train_vectors, pc_train['label'])


# In[ ]:


# Run predictions for PC Access logs

print(pc_clf.best_params_)

pc_test_scores = pc_clf.predict_proba(pc_test_vectors)[:,1]
pc_test_preds = pc_clf.predict(pc_test_vectors)


# In[ ]:


# Initialise and fit Multinomial NB for Building Access logs
# Scoring uses roc_auc

building_clf = GridSearchCV(MultinomialNB(),
                            param_grid=params,
                            scoring='roc_auc',
                            cv=cv,
                            verbose=3)

building_clf.fit(building_train_vectors, building_train['label'])


# In[ ]:


# Run predictions for Building Access logs

print(building_clf.best_params_)

building_test_scores = building_clf.predict_proba(building_test_vectors)[:,1]
building_test_preds = building_clf.predict(building_test_vectors)


# In[ ]:


# Initialise and fit Multinomial NB for Proxy logs
# Scoring uses roc_auc

proxy_clf = GridSearchCV(MultinomialNB(),
                         param_grid=params,
                         scoring='roc_auc',
                         cv=cv,
                         verbose=3)

proxy_clf.fit(proxy_train_vectors, proxy_train['label'])


# In[ ]:


# Run predictions for Proxy logs

print(proxy_clf.best_params_)

proxy_test_scores = proxy_clf.predict_proba(proxy_test_vectors)[:,1]
proxy_test_preds = proxy_clf.predict(proxy_test_vectors)


# ### Evaluate classifier

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[ ]:


cm_labels = [1, -1]
cm_display_labels = ['normal', 'anomaly']


# In[ ]:


# Compute metrics for PC classifier

pc_test_accuracy = accuracy_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_precision = precision_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_recall = recall_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_f1 = f1_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_f1_weighted = f1_score(pc_test['label'].tolist(), pc_test_preds, average='weighted')
pc_test_roc_auc = roc_auc_score(pc_test['label'].tolist(), pc_test_scores)
pc_test_confusion = confusion_matrix(pc_test['label'].tolist(), pc_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for PC classifier

print('Results for Multinomial NB on pc_test:')
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

building_test_accuracy = accuracy_score(building_test['label'].tolist(), building_test_preds)
building_test_precision = precision_score(building_test['label'].tolist(), building_test_preds)
building_test_recall = recall_score(building_test['label'].tolist(), building_test_preds)
building_test_f1 = f1_score(building_test['label'].tolist(), building_test_preds)
building_test_f1_weighted = f1_score(building_test['label'].tolist(), building_test_preds, average='weighted')
building_test_roc_auc = roc_auc_score(building_test['label'].tolist(), building_test_scores)
building_test_confusion = confusion_matrix(building_test['label'].tolist(), building_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for Building classifier

print('Results for Multinomial NB on building_test:')
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

proxy_test_accuracy = accuracy_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_precision = precision_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_recall = recall_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_f1 = f1_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_f1_weighted = f1_score(proxy_test['label'].tolist(), proxy_test_preds, average='weighted')
proxy_test_roc_auc = roc_auc_score(proxy_test['label'].tolist(), proxy_test_scores)
proxy_test_confusion = confusion_matrix(proxy_test['label'].tolist(), proxy_test_preds, labels=cm_labels)


# In[ ]:


# Display evaluation results for Proxy classifier

print('Results for Multinomial NB on proxy_test:')
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

path = 'models/mnb/'

if vectorizer == 'count':
    PC_CLF = path + 'pc_mnb_count.joblib'
    BUILDING_CLF = path + 'building_mnb_count.joblib'
    PROXY_CLF = path + 'proxy_mnb_count.joblib'
elif vectorizer == 'tfidf':
    PC_CLF = path + 'pc_mnb_tfidf.joblib'
    BUILDING_CLF = path + 'building_mnb_tfidf.joblib'
    PROXY_CLF = path + 'proxy_mnb_tfidf.joblib'

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

pc_unseen = pc_test[pc_test['suspect'] == 0].copy()

np.random.seed(480)
pc_unseen['machine_location'] = np.random.choice(countries, pc_unseen.shape[0])
pc_unseen['input'] = pc_unseen[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                                'machine_location', 'working_hours', 'user_location', 'terminated']].agg(', '.join, axis=1)

display(pc_unseen)


# In[ ]:


# Create new data for Building Access logs
# New case: large number of access attempts

building_unseen = building_test[building_test['suspect'] == 0].copy()

np.random.seed(480)
building_unseen['attempts'] = np.random.randint(6, 20, building_unseen.shape[0])
building_unseen['attempts'] = building_unseen['attempts'].astype(str)
building_unseen['input'] = building_unseen[['user_id', 'access_date_time', 'direction', 'status',
                                            'office_location', 'attempts', 'terminated']].agg(', '.join, axis=1)

display(building_unseen)


# In[ ]:


# Create new data for Proxy logs
# New case: data upload/download from malicious urls

url_types = ['malware', 'phishing']

malicious_urls_df = pd.read_csv('malicious_urls.csv', sep=',', header=0)
malicious_urls_df = malicious_urls_df[malicious_urls_df['type'].isin(url_types)]

display(malicious_urls_df)

urls = malicious_urls_df['url']

proxy_unseen = proxy_test[proxy_test['suspect'] == 0].copy()

np.random.seed(480)
proxy_unseen['url'] = np.random.choice(urls, proxy_unseen.shape[0])
proxy_unseen['category'] = 'Malware, Phishing'
proxy_unseen['input'] = proxy_unseen[['user_id', 'access_date_time', 'machine_name',
                                      'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)

display(proxy_unseen)


# In[ ]:


# Compute accuracy of PC classifier on unseen case

pc_unseen_vectors = pc_vectorizer.transform(pc_unseen['input'])
pc_unseen_scores = pc_clf_loaded.predict_proba(pc_unseen_vectors)[:,1]
pc_unseen_preds = pc_clf_loaded.predict(pc_unseen_vectors)

pc_results_counter = Counter(pc_unseen_preds)

print('PC classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[-1], len(pc_unseen_preds),
       pc_results_counter[-1] / len(pc_unseen_preds) * 100))


# In[ ]:


# Compute accuracy of Building classifier on unseen case

building_unseen_vectors = building_vectorizer.transform(building_unseen['input'])
building_unseen_scores = building_clf_loaded.predict_proba(building_unseen_vectors)[:,1]
building_unseen_preds = building_clf_loaded.predict(building_unseen_vectors)

building_results_counter = Counter(building_unseen_preds)

print('Building classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[-1], len(building_unseen_preds),
       building_results_counter[-1] / len(building_unseen_preds) * 100))


# In[ ]:


# Compute accuracy of Proxy classifier on unseen case

proxy_unseen_vectors = proxy_vectorizer.transform(proxy_unseen['input'])
proxy_unseen_scores = proxy_clf_loaded.predict_proba(proxy_unseen_vectors)[:,1]
proxy_unseen_preds = proxy_clf_loaded.predict(proxy_unseen_vectors)

proxy_results_counter = Counter(proxy_unseen_preds)

print('Proxy classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[-1], len(proxy_unseen_preds),
       proxy_results_counter[-1] / len(proxy_unseen_preds) * 100))


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
      (pc_results_counter[-1], len(pc_unseen_preds),
       pc_results_counter[-1] / len(pc_unseen_preds) * 100))

# Custom
pc_threshold = 0.95

pc_test_preds_manual = np.where(pc_test_scores < pc_threshold, -1, 1)
pc_test_confusion_manual = confusion_matrix(pc_test['label'].tolist(), pc_test_preds_manual, labels=cm_labels)
pc_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=pc_test_confusion_manual,
                                                       display_labels=cm_display_labels)

print('\nPC confusion matrix for test data (Custom threshold):')
pc_test_confusion_disp_manual.plot(values_format='')
plt.show()

pc_unseen_preds_manual = np.where(pc_unseen_scores < pc_threshold, -1, 1)
pc_results_counter_manual = Counter(pc_unseen_preds_manual)

print('PC classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter_manual[-1], len(pc_unseen_preds_manual),
       pc_results_counter_manual[-1] / len(pc_unseen_preds_manual) * 100))


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
      (building_results_counter[-1], len(building_unseen_preds),
       building_results_counter[-1] / len(building_unseen_preds) * 100))

# Custom
building_threshold = 0.965

building_test_preds_manual = np.where(building_test_scores < building_threshold, -1, 1)
building_test_confusion_manual = confusion_matrix(building_test['label'].tolist(), building_test_preds_manual, labels=cm_labels)
building_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=building_test_confusion_manual,
                                                             display_labels=cm_display_labels)

print('\nBuilding confusion matrix for test data (Custom threshold):')
building_test_confusion_disp_manual.plot(values_format='')
plt.show()

building_unseen_preds_manual = np.where(building_unseen_scores < building_threshold, -1, 1)
building_results_counter_manual = Counter(building_unseen_preds_manual)

print('Building classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter_manual[-1], len(building_unseen_preds_manual),
       building_results_counter_manual[-1] / len(building_unseen_preds_manual) * 100))


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
      (proxy_results_counter[-1], len(proxy_unseen_preds),
       proxy_results_counter[-1] / len(proxy_unseen_preds) * 100))

# Custom
proxy_threshold = 0.9937

proxy_test_preds_manual = np.where(proxy_test_scores < proxy_threshold, -1, 1)
proxy_test_confusion_manual = confusion_matrix(proxy_test['label'].tolist(), proxy_test_preds_manual, labels=cm_labels)
proxy_test_confusion_disp_manual = ConfusionMatrixDisplay(confusion_matrix=proxy_test_confusion_manual,
                                                          display_labels=cm_display_labels)

print('\nProxy confusion matrix for test data (Custom threshold):')
proxy_test_confusion_disp_manual.plot(values_format='')
plt.show()

proxy_unseen_preds_manual = np.where(proxy_unseen_scores < proxy_threshold, -1, 1)
proxy_results_counter_manual = Counter(proxy_unseen_preds_manual)

print('Proxy classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter_manual[-1], len(proxy_unseen_preds_manual),
       proxy_results_counter_manual[-1] / len(proxy_unseen_preds_manual) * 100))

