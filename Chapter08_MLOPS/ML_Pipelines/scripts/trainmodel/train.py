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

parser = argparse.ArgumentParser("train")

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
parser.add_argument("--model", type=str, help="model")

args = parser.parse_args()

run = Run.get_context()
clean_data = run.input_datasets['cleansed_data']
# get input dataset by name
data = clean_data.to_pandas_dataframe()
run.log("Training start time", str(datetime.datetime.now()))

# Model Training
X = data.drop(columns = 'deposit_cat')
y = data[['deposit_cat']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 300)

Params = {'learning_rate': np.float(args.learning_rate),
          'n_estimators': np.int(args.n_estimators),
          'max_depth': np.int(args.max_depth),
          'min_samples_split': np.int(args.min_samples_split),
          'min_samples_leaf': np.int(args.min_samples_leaf),
          'subsample': np.float(args.subsample),
          'random_state': np.int(args.random_state),
          'max_features': np.int(args.max_features)}

# GradientBoostingClassifier
model = GradientBoostingClassifier(**Params)
model.fit(X_train,y_train.squeeze().values)

#calculate and print scores for the model 
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(X_test)

model_file_name = 'joblibGB_bankmarketing.sav'

accuracy_score_train = accuracy_score(y_train, y_train_preds)
accuracy_score_test = accuracy_score(y_test, y_test_preds)
run.log('Gradient Boosting Accuracy Score for training', accuracy_score_train)
run.log('Graident Boosting Accuracy Score for testing', accuracy_score_test)

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/' + model_file_name)  

run.upload_file("outputs/joblibGB_bankmarketing.sav", "outputs/joblibGB_bankmarketing.sav")
model = run.register_model(model_name = 'bankmarketing_GBmodel_pipeline', model_path = 'outputs/joblibGB_bankmarketing.sav')
run.log("Training end time", str(datetime.datetime.now()))
run.complete()


