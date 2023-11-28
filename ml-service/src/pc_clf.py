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

    employee_df = pd.read_csv('employee_data_5k_new.csv', sep=';', header=0)
    pc_df = pd.read_csv('pc_data_5k_new.csv', sep=';', header=0)


    # Add 'terminated' column to Employee dataframe

    employee_df['terminated'] = np.where(employee_df['terminated_date'].notnull(), 'Y', 'N')


    # Inner join Employee dataframe with PC Access dataframe

    join_df = employee_df[['id', 'location', 'terminated']].rename(
        columns={'location': 'user_location'}
    )

    pc_df = pc_df.join(join_df.set_index('id'), on='user_id', how='inner')


    # Split 'access_date_time' column into multiple component columns

    datetime_format = '%Y-%m-%d %H:%M:%S'
    pc_df['access_date_time'] = pd.to_datetime(pc_df['access_date_time'], format=datetime_format)

    pc_df['access_year'] = pc_df['access_date_time'].map(lambda x: x.year)
    pc_df['access_month'] = pc_df['access_date_time'].map(lambda x: x.month)
    pc_df['access_day'] = pc_df['access_date_time'].map(lambda x: x.day)
    pc_df['access_weekday'] = pc_df['access_date_time'].map(lambda x: x.weekday())
    pc_df['access_hour'] = pc_df['access_date_time'].map(lambda x: x.hour)
    pc_df['access_minute'] = pc_df['access_date_time'].map(lambda x: x.minute)
    pc_df['access_second'] = pc_df['access_date_time'].map(lambda x: x.second)


    # Define function to assign binary truth labels for anomaly detection

    def label(input):
        if input == 0:
            return 1
        return -1


    # Add binary 'label' column

    pc_df['label'] = pc_df['suspect'].map(label)


    # Split data into train and test
    # 80% train, 20% test

    to_drop = ['id', 'user_id', 'access_date_time', 'machine_name', 'suspect']
    X_train, X_test, y_train, y_test = train_test_split(pc_df.drop(labels=to_drop, axis=1),
                                                        pc_df['label'],
                                                        test_size=0.2,
                                                        random_state=480)

    print('Data splits:')
    print(X_train.shape)
    print(X_test.shape)


    # Select only normal training data and drop 'label' column

    X_train_normal = X_train.loc[X_train['label'] == 1]
    X_train_normal = X_train_normal.drop(labels='label', axis=1)


    # Create feature encoding pipeline

    numeric_features = ['access_year']
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    categorical_features = ['log_on_off', 'machine_location', 'working_hours',
                            'user_location', 'terminated']
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

    dump(clf, 'models/pc_clf.joblib')



def infer(data):

    print('pc received:', data)

    # Load classifier

    clf_loaded = load('models/pc_clf.joblib')


    # Create new data for PC Access logs
    # New case: unseen machine location

    countries = ['Russia', 'China', 'India', 'Brazil']

    # unseen = X_test[X_test['label'] == 1].copy()
    unseen = pd.read_json(data)

    print(unseen)

    np.random.seed(480)
    unseen['machine_location'] = np.random.choice(countries, unseen.shape[0])


    # Compute accuracy of Local Outlier Factor on unseen case

    unseen_scores = clf_loaded.score_samples(unseen.drop(labels='label', axis=1))
    unseen_preds = clf_loaded.predict(unseen.drop(labels='label', axis=1))

    results_counter = Counter(unseen_preds)

    print('Local Outlier Factor accuracy on unseen case:')
    print('Correct predictions: %d/%d (%f%%)' %
        (results_counter[-1], len(unseen_preds), results_counter[-1] / len(unseen_preds) * 100))
    