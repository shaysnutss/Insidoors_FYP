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

pc_preprocessor = None



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
        employee_df = pd.read_sql('SELECT * FROM employees', con=con)
        pc_df = pd.read_sql('SELECT * FROM pc_access', con=con)


    # Add 'terminated' column to Employee dataframe

    employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


    # Inner join Employee dataframe with PC Access dataframe

    join_df = employee_df[['id', 'location', 'terminated']].rename(
        columns={'location': 'user_location'}
    )

    pc_df = pc_df.join(join_df.set_index('id'), on='user_id', how='inner')


    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Parameter grid for GridSearch hyperparameter tuning
    params = settings['params']

    # Number of folds for GridSearch cross-validation
    cv = settings['cv']


    # Split 'access_date_time' column into multiple component columns

    pc_df = split_date_time(pc_df)


    # Define function to assign binary truth labels for anomaly detection

    def label(input):
        if input == 0:
            return 1
        return -1


    # Add binary 'label' column

    pc_df['label'] = pc_df['suspect'].map(label)


    # Split PC dataframe into train and test
    # 80% train, 20% test

    pc_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'machine_lat', 'machine_long', 'suspect']
    pc_X_train, pc_X_test, pc_y_train, pc_y_test = train_test_split(pc_df.drop(labels=pc_drop, axis=1),
                                                                    pc_df['label'],
                                                                    test_size=0.2,
                                                                    random_state=480)


    # Select only normal pc_X_train data and drop 'label' column

    pc_X_train_normal = pc_X_train.loc[pc_X_train['label'] == 1]
    pc_X_train_normal = pc_X_train_normal.drop(labels='label', axis=1)


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

    global pc_preprocessor

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
    

    # Normalize encodings if StandardScaler is not used

    pc_test_encodings = pc_preprocessor.transform(pc_X_test.drop(labels='label', axis=1))

    if not useStandardScaler:        
        pc_train_encodings = normalize(pc_train_encodings)
        pc_test_encodings = normalize(pc_test_encodings)


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


    # Compute metrics for PC classifier

    pc_test_accuracy = accuracy_score(pc_y_test, pc_test_preds)
    pc_test_precision = precision_score(pc_y_test, pc_test_preds)
    pc_test_recall = recall_score(pc_y_test, pc_test_preds)
    pc_test_f1 = f1_score(pc_y_test, pc_test_preds)
    pc_test_f1_weighted = f1_score(pc_y_test, pc_test_preds, average='weighted')
    pc_test_roc_auc = roc_auc_score(pc_y_test, pc_test_scores)


    # Display evaluation results for PC classifier

    print('Results for Local Outlier Factor on pc_test:')
    print('Accuracy:', pc_test_accuracy)
    print('Precision:', pc_test_precision)
    print('Recall:', pc_test_recall)
    print('F1:', pc_test_f1)
    print('F1 (Weighted):', pc_test_f1_weighted)
    print('ROC AUC:', pc_test_roc_auc)


    # Save classifier

    if useStandardScaler:
        PC_CLF = '/models/pc_clf.joblib'
    else:
        PC_CLF = '/models/pc_clf.joblib'

    dump(pc_clf.best_estimator_, PC_CLF)



def classify(data):

    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Custom classifier threshold
    threshold = settings['pc_threshold']


    # Load classifier

    if useStandardScaler:
        pc_clf_loaded = load('/models/pc_clf.joblib')
    else:
        pc_clf_loaded = load('/models/pc_clf.joblib')


    # Preprocess data

    data = data.replace('/', '\\/')
    data = data.replace("'", '"')
    data_df = pd.read_json(StringIO(data))
    data_df = split_date_time(data_df)


    # Compute accuracy of PC classifier on incoming data

    data_encodings = pc_preprocessor.transform(data_df)
    if not useStandardScaler:
        data_encodings = normalize(data_encodings)

    data_scores = pc_clf_loaded.score_samples(data_encodings)
    data_preds = np.where(data_scores < threshold, -1, 1)
    counter = Counter(data_preds)

    print('PC classifier accuracy for unseen data (Custom threshold):')
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
