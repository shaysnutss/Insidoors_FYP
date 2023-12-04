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

building_preprocessor = None



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
        building_df = pd.read_sql('SELECT * FROM building_access', con=con)


    # Add 'terminated' column to Employee dataframe

    employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


    # Inner join Employee dataframe with Building Access dataframe

    join_df = employee_df[['id', 'terminated']]

    building_df = building_df.join(join_df.set_index('id'), on='user_id', how='inner')


    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Parameter grid for GridSearch hyperparameter tuning
    params = settings['params']

    # Number of folds for GridSearch cross-validation
    cv = settings['cv']


    # Split 'access_date_time' column into multiple component columns

    building_df = split_date_time(building_df)


    # Define function to assign binary truth labels for anomaly detection

    def label(input):
        if input == 0:
            return 1
        return -1


    # Add binary 'label' column

    building_df['label'] = building_df['suspect'].map(label)


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

    global building_preprocessor

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
    

    # Normalize encodings if StandardScaler is not used

    building_test_encodings = building_preprocessor.transform(building_X_test.drop(labels='label', axis=1))

    if not useStandardScaler:        
        building_train_encodings = normalize(building_train_encodings)
        building_test_encodings = normalize(building_test_encodings)


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


    # Compute metrics for Building classifier

    building_test_accuracy = accuracy_score(building_y_test, building_test_preds)
    building_test_precision = precision_score(building_y_test, building_test_preds)
    building_test_recall = recall_score(building_y_test, building_test_preds)
    building_test_f1 = f1_score(building_y_test, building_test_preds)
    building_test_f1_weighted = f1_score(building_y_test, building_test_preds, average='weighted')
    building_test_roc_auc = roc_auc_score(building_y_test, building_test_scores)


    # Display evaluation results for Building classifier

    print('Results for Local Outlier Factor on building_test:')
    print('Accuracy:', building_test_accuracy)
    print('Precision:', building_test_precision)
    print('Recall:', building_test_recall)
    print('F1:', building_test_f1)
    print('F1 (Weighted):', building_test_f1_weighted)
    print('ROC AUC:', building_test_roc_auc)


    # Save classifier

    if useStandardScaler:
        BUILDING_CLF = '/models/building_clf.joblib'
    else:
        BUILDING_CLF = '/models/building_clf.joblib'

    dump(building_clf.best_estimator_, BUILDING_CLF)



def classify(data):

    # Whether to use sklearn's StandardScaler() to scale numeric data
    useStandardScaler = settings['useStandardScaler']

    # Custom classifier threshold
    threshold = settings['building_threshold']


    # Load classifier

    if useStandardScaler:
        building_clf_loaded = load('/models/building_clf.joblib')
    else:
        building_clf_loaded = load('/models/building_clf.joblib')


    # Preprocess data

    data = data.replace('/', '\\/')
    data = data.replace("'", '"')
    data_df = pd.read_json(StringIO(data))
    data_df = split_date_time(data_df)


    # Compute accuracy of Building classifier on incoming data

    data_encodings = building_preprocessor.transform(data_df)
    if not useStandardScaler:
        data_encodings = normalize(data_encodings)

    data_scores = building_clf_loaded.score_samples(data_encodings)
    data_preds = np.where(data_scores < threshold, -1, 1)
    counter = Counter(data_preds)

    print('Building classifier accuracy for unseen data (Custom threshold):')
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
