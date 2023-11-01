#!/usr/bin/env python
# coding: utf-8

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

to_exclude = [2, 5]

pc_df = pc_df[~pc_df['suspect'].isin(to_exclude)]
building_df = building_df[~building_df['suspect'].isin(to_exclude)]
proxy_df = proxy_df[~proxy_df['suspect'].isin(to_exclude)]


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


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


# Concatenate input columns of PC Access logs into a single string and create label column

pc_df = pc_df.astype(str)
pc_df['suspect'] = pc_df['suspect'].astype(int) # Keep 'suspect' column as int
pc_df['input'] = pc_df[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                        'machine_location', 'user_location', 'terminated']].agg(', '.join, axis=1)
pc_df['label'] = pc_df['suspect']

display(pc_df)


# In[ ]:


# Prepare label column of PC Access dataframe for anomaly detection

pc_df.loc[pc_df['suspect'] == 0, 'label'] = 1
pc_df.loc[pc_df['suspect'] != 0, 'label'] = -1


# In[ ]:


# Concatenate input columns of Building Access logs into a single string and create label column

building_df = building_df.astype(str)
building_df['suspect'] = building_df['suspect'].astype(int) # Keep 'suspect' column as int
building_df['input'] = building_df[['user_id', 'access_date_time', 'direction', 'status',
                                    'office_location', 'attempts', 'terminated']].agg(', '.join, axis=1)
building_df['label'] = building_df['suspect']

display(building_df)


# In[ ]:


# Prepare label column of Building Access dataframe for anomaly detection

building_df.loc[building_df['suspect'] == 0, 'label'] = 1
building_df.loc[building_df['suspect'] != 0, 'label'] = -1


# In[ ]:


# Concatenate input columns of Proxy logs into a single string and create label column

proxy_df = proxy_df.astype(str)
proxy_df['suspect'] = proxy_df['suspect'].astype(int) # Keep 'suspect' column as int
proxy_df['input'] = proxy_df[['user_id', 'access_date_time', 'machine_name',
                              'url', 'category', 'bytes_in', 'bytes_out']].agg(', '.join, axis=1)
proxy_df['label'] = proxy_df['suspect']

display(proxy_df)


# In[ ]:


# Prepare label column of Proxy dataframe for anomaly detection

proxy_df.loc[proxy_df['suspect'] == 0, 'label'] = 1
proxy_df.loc[proxy_df['suspect'] != 0, 'label'] = -1


# In[ ]:


# Split PC dataset into train and test
# 80% train, 20% test

pc_train, pc_test = np.split(pc_df.sample(frac=1, random_state=480),
                             [int(0.8 * len(pc_df))])

print('PC dataframe splits:')
print(pc_train.shape)
print(pc_test.shape)


# In[ ]:


# Split Building dataset into train and test
# 80% train, 20% test

building_train, building_test = np.split(building_df.sample(frac=1, random_state=480),
                                        [int(0.8 * len(building_df))])

print('Building dataframe splits:')
print(building_train.shape)
print(building_test.shape)


# In[ ]:


# Split Proxy dataset into train and test
# 80% train, 20% test

proxy_train, proxy_test = np.split(proxy_df.sample(frac=1, random_state=480),
                                  [int(0.8 * len(proxy_df))])

print('Proxy dataframe splits:')
print(proxy_train.shape)
print(proxy_test.shape)


# In[ ]:


# Initialise tf-idf vectorizers

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


# ### Train anomaly detection classifier with One-class SVM

# In[ ]:


from sklearn.svm import OneClassSVM


# In[ ]:


# Initialise and fit One-class SVM classifiers

pc_clf = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
pc_clf.fit(pc_train_vectors)

building_clf = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
building_clf.fit(building_train_vectors)

proxy_clf = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
proxy_clf.fit(proxy_train_vectors)


# In[ ]:


# Run predictions for PC Access logs

pc_train_preds = pc_clf.predict(pc_train_vectors)
pc_test_preds = pc_clf.predict(pc_test_vectors)


# In[ ]:


# Run predictions for Building Access logs

building_train_preds = building_clf.predict(building_train_vectors)
building_test_preds = building_clf.predict(building_test_vectors)


# In[ ]:


# Run predictions for Proxy logs

proxy_train_preds = proxy_clf.predict(proxy_train_vectors)
proxy_test_preds = proxy_clf.predict(proxy_test_vectors)


# ### Evaluate classifier

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay


# In[ ]:


# Compute metrics for PC classifier

pc_train_accuracy = accuracy_score(pc_train['label'].tolist(), pc_train_preds)
pc_train_precision = precision_score(pc_train['label'].tolist(), pc_train_preds)
pc_train_recall = recall_score(pc_train['label'].tolist(), pc_train_preds)
pc_train_confusion = confusion_matrix(pc_train['label'].tolist(), pc_train_preds)

pc_test_accuracy = accuracy_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_precision = precision_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_recall = recall_score(pc_test['label'].tolist(), pc_test_preds)
pc_test_confusion = confusion_matrix(pc_test['label'].tolist(), pc_test_preds)


# In[ ]:


# Display evaluation results for pc_train

print('Results for One-class SVM on pc_train:')
print('Accuracy:', pc_train_accuracy)
print('Precision:', pc_train_precision)
print('Recall:', pc_train_recall)
print('Confusion Matrix: \n', pc_train_confusion)

# pc_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=pc_train_confusion,
#                                                  display_labels=pc_clf.classes_)
# pc_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for pc_test

print('Results for One-class SVM on pc_test:')
print('Accuracy:', pc_test_accuracy)
print('Precision:', pc_test_precision)
print('Recall:', pc_test_recall)
print('Confusion Matrix: \n', pc_test_confusion)

# pc_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=pc_test_confusion,
#                                                 display_labels=pc_clf.classes_)
# pc_test_confusion_disp.plot()
# plt.show()


# In[ ]:


# Compute metrics for Building classifier

building_train_accuracy = accuracy_score(building_train['label'].tolist(), building_train_preds)
building_train_precision = precision_score(building_train['label'].tolist(), building_train_preds)
building_train_recall = recall_score(building_train['label'].tolist(), building_train_preds)
building_train_confusion = confusion_matrix(building_train['label'].tolist(), building_train_preds)

building_test_accuracy = accuracy_score(building_test['label'].tolist(), building_test_preds)
building_test_precision = precision_score(building_test['label'].tolist(), building_test_preds)
building_test_recall = recall_score(building_test['label'].tolist(), building_test_preds)
building_test_confusion = confusion_matrix(building_test['label'].tolist(), building_test_preds)


# In[ ]:


# Display evaluation results for building_train

print('Results for One-class SVM on building_train:')
print('Accuracy:', building_train_accuracy)
print('Precision:', building_train_precision)
print('Recall:', building_train_recall)
print('Confusion Matrix: \n', building_train_confusion)

# building_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=building_train_confusion,
#                                                        display_labels=building_clf.classes_)
# building_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for building_test

print('Results for One-class SVM on building_test:')
print('Accuracy:', building_test_accuracy)
print('Precision:', building_test_precision)
print('Recall:', building_test_recall)
print('Confusion Matrix: \n', building_test_confusion)

# building_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=building_test_confusion,
#                                                       display_labels=building_clf.classes_)
# building_test_confusion_disp.plot()
# plt.show()


# In[ ]:


# Compute metrics for Proxy classifier

proxy_train_accuracy = accuracy_score(proxy_train['label'].tolist(), proxy_train_preds)
proxy_train_precision = precision_score(proxy_train['label'].tolist(), proxy_train_preds)
proxy_train_recall = recall_score(proxy_train['label'].tolist(), proxy_train_preds)
proxy_train_confusion = confusion_matrix(proxy_train['label'].tolist(), proxy_train_preds)

proxy_test_accuracy = accuracy_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_precision = precision_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_recall = recall_score(proxy_test['label'].tolist(), proxy_test_preds)
proxy_test_confusion = confusion_matrix(proxy_test['label'].tolist(), proxy_test_preds)


# In[ ]:


# Display evaluation results for proxy_train

print('Results for One-class SVM on proxy_train:')
print('Accuracy:', proxy_train_accuracy)
print('Precision:', proxy_train_precision)
print('Recall:', proxy_train_recall)
print('Confusion Matrix: \n', proxy_train_confusion)

# proxy_train_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=proxy_train_confusion,
#                                                     display_labels=proxy_clf.classes_)
# proxy_train_confusion_disp.plot()
# plt.show()


# In[ ]:


# Display evaluation results for proxy_test

print('Results for One-class SVM on proxy_test:')
print('Accuracy:', proxy_test_accuracy)
print('Precision:', proxy_test_precision)
print('Recall:', proxy_test_recall)
print('Confusion Matrix: \n', proxy_test_confusion)

# proxy_test_confusion_disp = ConfusionMatrixDisplay(confusion_matrix=proxy_test_confusion,
#                                                    display_labels=proxy_clf.classes_)
# proxy_test_confusion_disp.plot()
# plt.show()


# ### Test classifier predictions on unseen cases

# In[ ]:


from collections import Counter


# In[ ]:


# Create new data for PC Access logs
# New case: machine location different from user location

countries = ['Russia', 'China', 'India', 'Brazil']

pc_unseen = pc_test[pc_test['suspect'] == 0].copy()

np.random.seed(480)
pc_unseen['machine_location'] = np.random.choice(countries, pc_unseen.shape[0])
pc_unseen['input'] = pc_unseen[['user_id', 'access_date_time', 'log_on_off', 'machine_name',
                                      'machine_location', 'user_location', 'terminated']].agg(', '.join, axis=1)

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
pc_unseen_preds = pc_clf.predict(pc_unseen_vectors)

pc_results_counter = Counter(pc_unseen_preds)

print('PC classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[-1], len(pc_unseen_preds), pc_results_counter[-1]/len(pc_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Building classifier on unseen case

building_unseen_vectors = building_vectorizer.transform(building_unseen['input'])
building_unseen_preds = building_clf.predict(building_unseen_vectors)

building_results_counter = Counter(building_unseen_preds)

print('Building classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[-1], len(building_unseen_preds), building_results_counter[-1]/len(building_unseen_preds)*100))


# In[ ]:


# Compute accuracy of Proxy classifier on unseen case

proxy_unseen_vectors = proxy_vectorizer.transform(proxy_unseen['input'])
proxy_unseen_preds = proxy_clf.predict(proxy_unseen_vectors)

proxy_results_counter = Counter(proxy_unseen_preds)

print('Proxy classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[-1], len(proxy_unseen_preds), proxy_results_counter[-1]/len(proxy_unseen_preds)*100))

