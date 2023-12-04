import json
import os
from dotenv import load_dotenv
from io import StringIO

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.creation import CyclicalFeatures

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV

from joblib import dump, load

from collections import Counter


# Global variables

with open('settings.json', 'r') as file:
    settings = json.load(file)

proxy_preprocessor = None



def train():

    # Create connection to RDS

    load_dotenv()

    conf = {
        'host': os.getenv('RDS_HOST'),
        'port': os.getenv('RDS_PORT'),
        'database': os.getenv('RDS_DB_TRAIN'),
        'user': os.getenv('RDS_USER'),
        'password': os.getenv('RDS_PASSWORD'),
    }

    engine = create_engine("mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**conf))


    # Load data from database into dataframes

    with engine.connect() as con:
        proxy_df = pd.read_sql('SELECT * FROM proxy_log', con=con)


    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Parameter grid for GridSearch hyperparameter tuning
    params = settings['params']

    # Number of folds for GridSearch cross-validation
    cv = settings['cv']


    # Split 'access_date_time' column into multiple component columns

    proxy_df = split_date_time(proxy_df)


    # Define function to assign binary truth labels for anomaly detection

    def label(input):
        if input == 0:
            return 1
        return -1


    # Add binary 'label' column

    proxy_df['label'] = proxy_df['suspect'].map(label)


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

    global proxy_preprocessor

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
    

    # Normalize encodings if StandardScaler is not used

    proxy_test_encodings = proxy_preprocessor.transform(proxy_X_test.drop(labels='label', axis=1))

    if not useStandardScaler:        
        proxy_train_encodings = normalize(proxy_train_encodings)
        proxy_test_encodings = normalize(proxy_test_encodings)


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

    proxy_test_scores = proxy_clf.score_samples(proxy_test_encodings)
    proxy_test_preds = proxy_clf.predict(proxy_test_encodings)


    # Compute metrics for Proxy classifier

    proxy_test_accuracy = accuracy_score(proxy_y_test, proxy_test_preds)
    proxy_test_precision = precision_score(proxy_y_test, proxy_test_preds)
    proxy_test_recall = recall_score(proxy_y_test, proxy_test_preds)
    proxy_test_f1 = f1_score(proxy_y_test, proxy_test_preds)
    proxy_test_f1_weighted = f1_score(proxy_y_test, proxy_test_preds, average='weighted')
    proxy_test_roc_auc = roc_auc_score(proxy_y_test, proxy_test_scores)


    # Display evaluation results for Proxy classifier

    print('Results for Local Outlier Factor on proxy_test:')
    print('Accuracy:', proxy_test_accuracy)
    print('Precision:', proxy_test_precision)
    print('Recall:', proxy_test_recall)
    print('F1:', proxy_test_f1)
    print('F1 (Weighted):', proxy_test_f1_weighted)
    print('ROC AUC:', proxy_test_roc_auc)


    # Save classifier

    if useStandardScaler:
        PROXY_CLF = '/models/proxy_clf.joblib'
    else:
        PROXY_CLF = '/models/proxy_clf.joblib'

    dump(proxy_clf.best_estimator_, PROXY_CLF)



def classify(data):

    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Custom classifier threshold
    threshold = settings['proxy_threshold']


    # Load classifier

    if useStandardScaler:
        proxy_clf_loaded = load('/models/proxy_clf.joblib')
    else:
        proxy_clf_loaded = load('/models/proxy_clf.joblib')


    # Preprocess data

    data = data.replace('/', '\\/')
    data = data.replace("'", '"')
    data_df = pd.read_json(StringIO(data))
    data_df = split_date_time(data_df)


    # Compute accuracy of Proxy classifier on incoming data

    data_encodings = proxy_preprocessor.transform(data_df)
    if not useStandardScaler:
        data_encodings = normalize(data_encodings)

    data_scores = proxy_clf_loaded.score_samples(data_encodings)
    data_preds = np.where(data_scores < threshold, -1, 1)
    counter = Counter(data_preds)

    print('Proxy classifier accuracy for unseen data (Custom threshold):')
    print('Correct predictions: %d/%d (%f%%)' %
        (counter[-1], len(data_preds), counter[-1] / len(data_preds) * 100))    


    # Append prediction results to dataframe

    data_df['suspect'] = data_preds
    data_df = data_df[['id', 'user_id']].loc[data_df['suspect'] == -1]


    # Return dataframe as json to rule-based algorithm controller

    return data_df.to_json(orient='records')



# Define function to split 'access_date_time' column into multiple component columns

def split_date_time(df):

    datetime_format = '%Y-%m-%d %H:%M:%S'
    df['access_date_time'] = pd.to_datetime(df['access_date_time'], format=datetime_format)

    df['access_year'] = df['access_date_time'].map(lambda x: x.year)
    df['access_month'] = df['access_date_time'].map(lambda x: x.month)
    df['access_day'] = df['access_date_time'].map(lambda x: x.day)
    df['access_weekday'] = df['access_date_time'].map(lambda x: x.weekday())
    df['access_hour'] = df['access_date_time'].map(lambda x: x.hour)
    df['access_minute'] = df['access_date_time'].map(lambda x: x.minute)
    df['access_second'] = df['access_date_time'].map(lambda x: x.second)

    return df



# Define function to normalize encodings

def normalize(x):

    m_ = np.mean(x, axis=1, keepdims=True)
    x = x - m_
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    return x
