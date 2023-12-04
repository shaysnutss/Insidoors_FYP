import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.creation import CyclicalFeatures

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from joblib import dump, load

from collections import Counter


# Create connection to RDS

conf = {
    'host':'master-data.ctk1qxl1tufb.ap-southeast-1.rds.amazonaws.com',
    'port':'3306',
    'database': 'training-data',
    'user': 'admin',
    'password': 'password'
}

engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**conf))


# Load data from database into dataframes

with engine.connect() as con:
    employee_df = pd.read_sql('SELECT * FROM employees', con=con)
    pc_df = pd.read_sql('SELECT * FROM pc_access', con=con)
    building_df = pd.read_sql('SELECT * FROM building_access', con=con)
    proxy_df = pd.read_sql('SELECT * FROM proxy_log', con=con)


# Add 'terminated' column to Employee dataframe

employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


# Inner join Employee dataframe with PC Access and Building Access dataframes

join_df_pc = employee_df[['id', 'location', 'terminated']].rename(
    columns={'location': 'user_location'}
)
join_df_building = employee_df[['id', 'terminated']]

pc_df = pc_df.join(join_df_pc.set_index('id'), on='user_id', how='inner')
building_df = building_df.join(join_df_building.set_index('id'), on='user_id', how='inner')


# Whether to use sklearn's StandardScaler() to scale numeric data
useStandardScaler = False

# Parameter grid for GridSearch hyperparameter tuning
params = {
    'n_neighbors': [20, 25, 30],
    'contamination': ['auto', 0.025, 0.05]
}

# Number of folds for GridSearch cross-validation
cv = 3


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


# Define function to assign binary truth labels for anomaly detection

def label(input):
    if input == 0:
        return 1
    return -1


# Add binary 'label' column to PC Access dataframe

pc_df['label'] = pc_df['suspect'].map(label)


# Add binary 'label' column to Building Access dataframe

building_df['label'] = building_df['suspect'].map(label)


# Add binary 'label' column to Proxy dataframe

proxy_df['label'] = proxy_df['suspect'].map(label)


# Split PC Access dataframe into train and test
# 80% train, 20% test

pc_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'machine_lat', 'machine_long', 'suspect']
pc_X_train, pc_X_test, pc_y_train, pc_y_test = train_test_split(pc_df.drop(labels=pc_drop, axis=1),
                                                                pc_df['label'],
                                                                test_size=0.2,
                                                                random_state=480)


# Select only normal pc_X_train data and drop 'label' column

pc_X_train_normal = pc_X_train.loc[pc_X_train['label'] == 1]
pc_X_train_normal = pc_X_train_normal.drop(labels='label', axis=1)


# Split Building dataframe into train and test
# 80% train, 20% test

building_drop = ['id', 'user_id', 'access_date_time', 'office_lat', 'office_long', 'suspect']
building_X_train, building_X_test, building_y_train, building_y_test = train_test_split(building_df.drop(labels=building_drop, axis=1),
                                                                                        building_df['label'],
                                                                                        test_size=0.2,
                                                                                        random_state=480)


# Select only normal building_X_train data and drop 'label' column

building_X_train_normal = building_X_train.loc[building_X_train['label'] == 1]
building_X_train_normal = building_X_train_normal.drop(labels='label', axis=1)


# Split Proxy dataframe into train and test
# 80% train, 20% test

proxy_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'url', 'suspect']
proxy_X_train, proxy_X_test, proxy_y_train, proxy_y_test = train_test_split(proxy_df.drop(labels=proxy_drop, axis=1),
                                                                            proxy_df['label'],
                                                                            test_size=0.2,
                                                                            random_state=480)


# Select only normal proxy_X_train data and drop 'label' column

proxy_X_train_normal = proxy_X_train.loc[proxy_X_train['label'] == 1]
proxy_X_train_normal = proxy_X_train_normal.drop(labels='label', axis=1)


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

pc_preprocessor.fit(pc_X_train_normal)

pc_train_encodings = pc_preprocessor.transform(pc_X_train_normal)


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

building_preprocessor.fit(building_X_train_normal)

building_train_encodings = building_preprocessor.transform(building_X_train_normal)


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

proxy_preprocessor.fit(proxy_X_train_normal)

proxy_train_encodings = proxy_preprocessor.transform(proxy_X_train_normal)


# Define function to normalize encodings

def normalize(x):
    m_ = np.mean(x, axis=1, keepdims=True)
    x = x - m_
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    return x


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


# Define GridSearch scoring function for PC Access logs
# Scoring uses roc_auc

def pc_scorer(estimator, _):
    scores = estimator.score_samples(pc_test_encodings)
    return roc_auc_score(pc_y_test, scores)


# Initialise and fit Local Outlier Factor for PC Access logs

pc_clf = GridSearchCV(LocalOutlierFactor(novelty=True),
                      param_grid=params,
                      scoring=pc_scorer,
                      cv=cv,
                      verbose=3)

pc_clf.fit(pc_train_encodings)


# Run predictions for PC Access logs

pc_test_scores = pc_clf.score_samples(pc_test_encodings)
pc_test_preds = pc_clf.predict(pc_test_encodings)


# Define GridSearch scoring function for Building Access logs
# Scoring uses roc_auc

def building_scorer(estimator, _):
    scores = estimator.score_samples(building_test_encodings)
    return roc_auc_score(building_y_test, scores)


# Initialise and fit Local Outlier Factor for Building Access logs

building_clf = GridSearchCV(LocalOutlierFactor(novelty=True),
                            param_grid=params,
                            scoring=building_scorer,
                            cv=cv,
                            verbose=3)

building_clf.fit(building_train_encodings)


# Run predictions for Building Access logs

building_test_scores = building_clf.score_samples(building_test_encodings)
building_test_preds = building_clf.predict(building_test_encodings)


# Define GridSearch scoring function for Proxy logs
# Scoring uses roc_auc

def proxy_scorer(estimator, _):
    scores = estimator.score_samples(proxy_test_encodings)
    return roc_auc_score(proxy_y_test, scores)


# Initialise and fit Local Outlier Factor for Proxy logs

proxy_clf = GridSearchCV(LocalOutlierFactor(novelty=True),
                         param_grid=params,
                         scoring=proxy_scorer,
                         cv=cv,
                         verbose=3)

proxy_clf.fit(proxy_train_encodings)


# Run predictions for Proxy logs

print(proxy_clf.best_params_)

proxy_test_scores = proxy_clf.score_samples(proxy_test_encodings)
proxy_test_preds = proxy_clf.predict(proxy_test_encodings)


# Compute metrics for PC classifier

cm_labels = [1, -1]
cm_display_labels = ['normal', 'anomaly']

pc_test_accuracy = accuracy_score(pc_y_test, pc_test_preds)
pc_test_precision = precision_score(pc_y_test, pc_test_preds)
pc_test_recall = recall_score(pc_y_test, pc_test_preds)
pc_test_f1 = f1_score(pc_y_test, pc_test_preds)
pc_test_f1_weighted = f1_score(pc_y_test, pc_test_preds, average='weighted')
pc_test_roc_auc = roc_auc_score(pc_y_test, pc_test_scores)
pc_test_confusion = confusion_matrix(pc_y_test, pc_test_preds, labels=cm_labels)


# Compute metrics for Building classifier

building_test_accuracy = accuracy_score(building_y_test, building_test_preds)
building_test_precision = precision_score(building_y_test, building_test_preds)
building_test_recall = recall_score(building_y_test, building_test_preds)
building_test_f1 = f1_score(building_y_test, building_test_preds)
building_test_f1_weighted = f1_score(building_y_test, building_test_preds, average='weighted')
building_test_roc_auc = roc_auc_score(building_y_test, building_test_scores)
building_test_confusion = confusion_matrix(building_y_test, building_test_preds, labels=cm_labels)


# Compute metrics for Proxy classifier

proxy_test_accuracy = accuracy_score(proxy_y_test, proxy_test_preds)
proxy_test_precision = precision_score(proxy_y_test, proxy_test_preds)
proxy_test_recall = recall_score(proxy_y_test, proxy_test_preds)
proxy_test_f1 = f1_score(proxy_y_test, proxy_test_preds)
proxy_test_f1_weighted = f1_score(proxy_y_test, proxy_test_preds, average='weighted')
proxy_test_roc_auc = roc_auc_score(proxy_y_test, proxy_test_scores)
proxy_test_confusion = confusion_matrix(proxy_y_test, proxy_test_preds, labels=cm_labels)


# Save classifiers

path = 'models/lof/'

if useStandardScaler:
    PC_CLF = path + 'pc_lof_scale.joblib'
    BUILDING_CLF = path + 'building_lof_scale.joblib'
    PROXY_CLF = path + 'proxy_lof_scale.joblib'
else:
    PC_CLF = path + 'pc_lof_norm.joblib'
    BUILDING_CLF = path + 'building_lof_norm.joblib'
    PROXY_CLF = path + 'proxy_lof_norm.joblib'

dump(pc_clf, PC_CLF)
dump(building_clf, BUILDING_CLF)
dump(proxy_clf, PROXY_CLF)


# Load classifiers

pc_clf_loaded = load(PC_CLF)
building_clf_loaded = load(BUILDING_CLF)
proxy_clf_loaded = load(PROXY_CLF)


# Create new data for PC Access logs
# New case: unseen machine location

countries = ['Russia', 'China', 'India', 'Brazil']

pc_unseen = pc_X_test[pc_X_test['label'] == 1].copy()

np.random.seed(480)
pc_unseen['machine_location'] = np.random.choice(countries, pc_unseen.shape[0])


# Create new data for Building Access logs
# New case: large number of access attempts

building_unseen = building_X_test[building_X_test['label'] == 1].copy()

np.random.seed(480)
building_unseen['attempts'] = np.random.randint(6, 20, building_unseen.shape[0])


# Create new data for Proxy logs
# New case: data upload/download from malicious urls

url_types = ['malware', 'phishing']

malicious_urls_df = pd.read_csv('malicious_urls.csv', sep=',', header=0)
malicious_urls_df = malicious_urls_df[malicious_urls_df['type'].isin(url_types)]

urls = malicious_urls_df['url']

proxy_unseen = proxy_X_test[proxy_X_test['label'] == 1].copy()

np.random.seed(480)
# proxy_unseen['url'] = np.random.choice(urls, proxy_unseen.shape[0])
proxy_unseen['category'] = 'Malware, Phishing'


# Compute accuracy of PC classifier on unseen case

pc_unseen_encodings = pc_preprocessor.transform(pc_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    pc_unseen_encodings = normalize(pc_unseen_encodings)

pc_unseen_scores = pc_clf_loaded.score_samples(pc_unseen_encodings)
pc_unseen_preds = pc_clf_loaded.predict(pc_unseen_encodings)

pc_results_counter = Counter(pc_unseen_preds)

print('PC classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter[-1], len(pc_unseen_preds),
       pc_results_counter[-1] / len(pc_unseen_preds) * 100))


# Compute accuracy of Building classifier on unseen case

building_unseen_encodings = building_preprocessor.transform(building_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    building_unseen_encodings = normalize(building_unseen_encodings)

building_unseen_scores = building_clf_loaded.score_samples(building_unseen_encodings)
building_unseen_preds = building_clf_loaded.predict(building_unseen_encodings)

building_results_counter = Counter(building_unseen_preds)

print('Building classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter[-1], len(building_unseen_preds),
       building_results_counter[-1] / len(building_unseen_preds) * 100))


# Compute accuracy of Proxy classifier on unseen case

proxy_unseen_encodings = proxy_preprocessor.transform(proxy_unseen.drop(labels='label', axis=1))
if not useStandardScaler:
    proxy_unseen_encodings = normalize(proxy_unseen_encodings)

proxy_unseen_scores = proxy_clf_loaded.score_samples(proxy_unseen_encodings)
proxy_unseen_preds = proxy_clf_loaded.predict(proxy_unseen_encodings)

proxy_results_counter = Counter(proxy_unseen_preds)

print('Proxy classifier accuracy on unseen case:')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter[-1], len(proxy_unseen_preds),
       proxy_results_counter[-1] / len(proxy_unseen_preds) * 100))


# Manually set custom threshold for PC classifier

pc_threshold = -1.17

pc_unseen_preds_manual = np.where(pc_unseen_scores < pc_threshold, -1, 1)
pc_results_counter_manual = Counter(pc_unseen_preds_manual)

print('PC classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (pc_results_counter_manual[-1], len(pc_unseen_preds_manual),
       pc_results_counter_manual[-1] / len(pc_unseen_preds_manual) * 100))


# Manually set custom threshold for Building classifier

building_threshold = -1.164

building_unseen_preds_manual = np.where(building_unseen_scores < building_threshold, -1, 1)
building_results_counter_manual = Counter(building_unseen_preds_manual)

print('Building classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (building_results_counter_manual[-1], len(building_unseen_preds_manual),
       building_results_counter_manual[-1] / len(building_unseen_preds_manual) * 100))


# Manually set custom threshold for Proxy classifier

proxy_threshold = -1.62

proxy_unseen_preds_manual = np.where(proxy_unseen_scores < proxy_threshold, -1, 1)
proxy_results_counter_manual = Counter(proxy_unseen_preds_manual)

print('Proxy classifier accuracy for unseen data (Custom threshold):')
print('Correct predictions: %d/%d (%f%%)' %
      (proxy_results_counter_manual[-1], len(proxy_unseen_preds_manual),
       proxy_results_counter_manual[-1] / len(proxy_unseen_preds_manual) * 100))

