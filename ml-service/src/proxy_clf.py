from io import StringIO
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from feature_engine.creation import CyclicalFeatures

from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from joblib import dump, load

from collections import Counter


def train():

    # Load data

    proxy_df = pd.read_csv('proxy_data_5k_new.csv', sep=';', header=0)


    # Split 'access_date_time' column into multiple component columns

    proxy_df = split_date_time(proxy_df)


    # Define function to assign binary truth labels for anomaly detection

    def label(input):
        if input == 0:
            return 1
        return -1


    # Add binary 'label' column

    proxy_df['label'] = proxy_df['suspect'].map(label)


    # Split data into train and test
    # 80% train, 20% test

    to_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'url', 'suspect']
    X_train, X_test, y_train, y_test = train_test_split(proxy_df.drop(labels=to_drop, axis=1),
                                                        proxy_df['label'],
                                                        test_size=0.2,
                                                        random_state=480)

    print('Data splits:')
    print(X_train.shape)
    print(X_test.shape)


    # Select only normal training data and drop 'label' column

    X_train_normal = X_train.loc[X_train['label'] == 1]
    X_train_normal = X_train_normal.drop(labels='label', axis=1)


    # Create feature encoding pipeline

    numeric_features = ['bytes_in', 'bytes_out', 'access_year']
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    categorical_features = ['category']
    categorical_transformer = Pipeline(
        steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
    )

    cyclical_features = ['access_month', 'access_day', 'access_weekday',
                        'access_hour', 'access_minute', 'access_second']
    cyclical_transformer = Pipeline(
        steps=[('encoder', CyclicalFeatures(drop_original=True))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features),
            ('cyclical', cyclical_transformer, cyclical_features)
        ],
        remainder='drop'
    )


    # Initialise and fit Local Outlier Factor

    clf = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LocalOutlierFactor(novelty=True))
        ]
    )

    clf.fit(X_train_normal)


    # Run predictions

    test_scores = clf.score_samples(X_test.drop(labels='label', axis=1))
    test_preds = clf.predict(X_test.drop(labels='label', axis=1))


    # Compute metrics

    cm_labels = [1, -1]

    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_f1_weighted = f1_score(y_test, test_preds, average='weighted')
    test_roc_auc = roc_auc_score(y_test, test_scores)
    test_confusion = confusion_matrix(y_test, test_preds, labels=cm_labels)


    # Display evaluation results

    print('Results for Local Outlier Factor:')
    print('Accuracy:', test_accuracy)
    print('Precision:', test_precision)
    print('Recall:', test_recall)
    print('F1:', test_f1)
    print('F1 (Weighted):', test_f1_weighted)
    print('ROC AUC:', test_roc_auc)
    print('Confusion Matrix:\n', test_confusion)


    # Save classifier

    dump(clf, 'models/proxy_clf.joblib')



def infer(data):

    print('proxy received:', data)

    # Load classifier

    clf_loaded = load('models/proxy_clf.joblib')


    # Create new data for Proxy logs
    # New case: data upload/download from malicious urls

    # url_types = ['malware', 'phishing']

    # malicious_urls_df = pd.read_csv('malicious_urls.csv', sep=',', header=0)
    # malicious_urls_df = malicious_urls_df[malicious_urls_df['type'].isin(url_types)]

    # unseen = X_test[X_test['label'] == 1].copy()
    data = data.replace('/', '\\/')
    data = data.replace("'", '"')
    unseen = pd.read_json(StringIO(data))
    unseen = split_date_time(unseen)

    # print('columns:', unseen.columns)
    # print('df:', unseen)

    # urls = malicious_urls_df['url']
    # np.random.seed(480)
    # unseen['url'] = np.random.choice(urls, unseen.shape[0])

    # unseen['category'] = 'Malware, Phishing'


    # Compute accuracy of Local Outlier Factor on unseen case

    unseen_scores = clf_loaded.score_samples(unseen.drop(labels=['suspect', 'access_date_time'], axis=1))
    unseen_preds = clf_loaded.predict(unseen.drop(labels=['suspect', 'access_date_time'], axis=1))

    results_counter = Counter(unseen_preds)

    print('Local Outlier Factor accuracy on unseen case:')
    print('Correct predictions: %d/%d (%f%%)' %
        (results_counter[-1], len(unseen_preds), results_counter[-1] / len(unseen_preds) * 100))


    # Append prediction results to dataframe

    unseen['suspect'] = unseen_preds
    unseen = unseen[['id', 'user_id']].loc[unseen['suspect'] == -1]


    # Return dataframe as json to rule-based algorithm controller

    return unseen.to_json(orient='records')



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
