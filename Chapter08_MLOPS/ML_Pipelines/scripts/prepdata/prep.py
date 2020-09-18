import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from azureml.core import Dataset, Run
from sklearn import __version__ as sklearnver
from packaging.version import Version
if Version(sklearnver) < Version("0.23.0"):
    from sklearn.externals import joblib
else:
    import joblib

print("Cleans the input data")
run = Run.get_context()
# get input dataset by name
run.log("data cleaning start time", str(datetime.datetime.now()))
bank_dataset = run.input_datasets['bank_dataset']
 
parser = argparse.ArgumentParser("prep")
parser.add_argument("--output_cleanse", type=str, help="cleaned and transformed bank marketing data directory")
args = parser.parse_args()
print("Argument (output cleansed bank marketing data path): %s" % args.output_cleanse)
 
#to pandas dataframe
data = bank_dataset.to_pandas_dataframe()

# Data Cleaning
cat_col = ['default', 'housing', 'loan', 'deposit', 'job', 
            'marital', 'education', 'contact', 'month', 'poutcome']
for column in cat_col:
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(data[column])
    label_encoded_y = label_encoder.transform(data[column])
    data[column + '_cat'] = label_encoded_y
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

if not (args.output_cleanse is None):
    os.makedirs(args.output_cleanse, exist_ok=True)
    print("%s created" % args.output_cleanse)
    path = args.output_cleanse + "/processed.parquet"
    write_df = data.to_parquet(path)
run.log("data cleaning end time", str(datetime.datetime.now()))

