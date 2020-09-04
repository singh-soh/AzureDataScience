
import os
import argparse
import pickle
import pandas as pd
from azureml.core import Dataset, Run
import numpy as np
from sklearn.metrics import accuracy_score #metrics
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# sklearn.externals.joblib is removed in 0.23
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.23.0"):
    from sklearn.externals import joblib
else:
    import joblib
    
run = Run.get_context()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning_rate parameter to be used in the algorithm')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='n_estimators to be used in the algorithm')
    parser.add_argument('--max_depth', type=int, default=3,
                        help='max_depth parameter to be used in the algorithm')
    parser.add_argument('--min_samples_split', type=int, default=100,
                        help='min_samples_split to be used in the algorithm')
    parser.add_argument('--min_samples_leaf', type=int, default=100,
                        help='min_samples_leaf to be used in the algorithm')
    parser.add_argument('--subsample', type=float, default=3,
                        help='subsample parameter to be used in the algorithm')
    parser.add_argument('--random_state', type=int, default=0.7,
                        help='random_state to be used in the algorithm')
    parser.add_argument('--max_features', type=int, default=0.0,
                        help='max_features parameter to be used in the algorithm')
    

    args = parser.parse_args()  
    run.log('learning_rate', np.float(args.learning_rate))
    run.log('n_estimators', np.int(args.n_estimators))
    run.log('max_depth', np.int(args.max_depth))
    run.log('min_samples_split', np.int(args.min_samples_split))
    run.log('min_samples_leaf', np.int(args.min_samples_leaf))
    run.log('subsample', np.float(args.subsample))
    run.log('random_state', np.int(args.subsample))
    run.log('max_features', np.int(args.subsample))


    # get input dataset by name
    bank_dataset = run.input_datasets['bank_dataset']
    data = bank_dataset.to_pandas_dataframe()

    
    # Data Cleaning
    cat_col = ['default', 'housing', 'loan', 'deposit', 'job', 
                'marital', 'education', 'contact', 'month', 'poutcome']
    for column in cat_col:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(data[column])
        label_encoded_y = label_encoder.transform(data[column])
        data[column + '_cat'] = label_encoded_y
    #     data[column + '_bool'] = data[column].eq('yes').mul(1)
    data = data.drop(columns = cat_col)
    
    #drop irrelevant columns
    data = data.drop(columns = ['pdays'])
    #impute incorrect values and drop original columns
    def get_correct_values(row, column_name, threshold, df):
        ''' Returns mean value if value in column_name is above threshold'''
        if row[column_name] <= threshold:
            return row[column_name]
        else:
            mean = df[df[column_name] <= threshold][column_name].mean()
            return mean
    data['campaign_cleaned'] = data.apply(lambda row: get_correct_values(row, 'campaign', 50, data),axis=1)
    data['previous_cleaned'] = data.apply(lambda row: get_correct_values(row, 'previous', 50, data),axis=1)
    data = data.drop(columns = ['campaign', 'previous'])


    # Model Training
    X = data.drop(columns = 'deposit_cat')
    y = data[['deposit_cat']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
    Params = {'learning_rate': np.float(args.learning_rate),
              'n_estimators': np.int(args.n_estimators),
              'max_depth': np.int(args.max_depth),
              'min_samples_split': np.int(args.min_samples_split),
              'min_samples_leaf': np.int(args.min_samples_leaf),
              'subsample': np.float(args.subsample),
              'random_state': np.int(args.random_state),
              'max_features': np.int(args.max_features)}
        
    # GradientBoostingClassifier
    clf = GradientBoostingClassifier(**Params)
    clf.fit(X_train,y_train.squeeze().values)
    
    #calculate and print scores for the model 
    y_train_preds = clf.predict(X_train)
    y_test_preds = clf.predict(X_test)


    model_file_name = 'joblibGB_bankmarketing.sav'

    accuracy_score_train = accuracy_score(y_train, y_train_preds)
    accuracy_score_test = accuracy_score(y_test, y_test_preds)
    run.log('Gradient Boosting Accuracy Score for training', accuracy_score_train)
    run.log('Graident Boosting Accuracy Score for testing', accuracy_score_test)

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=clf, filename='outputs/' + model_file_name)    

if __name__ == '__main__':
    main()
