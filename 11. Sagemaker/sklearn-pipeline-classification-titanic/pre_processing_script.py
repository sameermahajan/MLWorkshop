from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder


col_to_predict = 'survived'
col_primary_identifer = "user_id"

COLLIST_META = [col_primary_identifer]
COLLIST_FEATURE = ['age', 'fare', 'embarked', 'sex', 'pclass']
COLLIST_ALL     = [col_to_predict] + COLLIST_META + COLLIST_FEATURE
COLLIST_NUMERIC = ['age', 'fare']
COLLIST_CATEGORICAL = ['embarked', 'sex', 'pclass']
# pclass is int BUT ordinal


col_to_predict_dtype = {col_to_predict: "int64"} 

COLLIST_FEATURE_DTYPE = {
    'age': "float64",
    'fare': "float64",
    'embarked': "category",
    'sex': "category",
    'pclass': "int64"
}

COLLIST_META_DTYPE = {
    "user_id": "object"
}


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


if __name__ == '__main__':
    
    """
    1. read the raw data
    2. pre-process it i.e fit sklearn preprocess pipeline on it
    3. PERSIST THE TRAINED preprocess MODEL
    4. exit
    
    In nutshell just get a trained sklearn-pre-processor
    """
    print("{}_sklearn-pre-processor_{}".format("="*40, "="*40))
    
    # step_1 : parse the command line inputs
    print('extracting arguments')
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    
    
    # step_2 : load raw df from args.train path
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    dtype_dict = merge_two_dicts(COLLIST_FEATURE_DTYPE, COLLIST_META_DTYPE)
    dtype_dict = merge_two_dicts(COLLIST_FEATURE_DTYPE, col_to_predict_dtype)
    read_columns = COLLIST_ALL
    raw_data = [ pd.read_csv(file, header=None, names=read_columns, dtype=dtype_dict) for file in input_files ]
    concat_data = pd.concat(raw_data)
    concat_data = concat_data[COLLIST_ALL]
    
    print("data loading completed:")
    print("data shape : ", concat_data.shape)
    print("columns : {}".format(COLLIST_ALL))
    print("loaded RAW data : \n", concat_data.head(1).values, "\n")
    arr = concat_data[col_to_predict].value_counts()
    print("to_predict_col : {} : {}\n".format(arr.index.values, arr.values))

    # only pass feature cols, abels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
    concat_data = concat_data[COLLIST_FEATURE]
    
    # step_3 : pre-processor
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, COLLIST_NUMERIC),
            ('cat', categorical_transformer, COLLIST_CATEGORICAL)], 
        remainder='drop',
        verbose=True)
    
    # step_4 : fit the pre-processor
    print("before pp : FEATURE data shape : ", concat_data.shape)
    print("columns ({} columns): {}".format(len(COLLIST_FEATURE), COLLIST_FEATURE))
    print("sample data : \n", concat_data[COLLIST_NUMERIC+COLLIST_CATEGORICAL].head(1).values, "\n")
    preprocessor.fit(concat_data)
    pp_data = preprocessor.transform(concat_data) # return numpy array
    
    # below code to get colnames
    enc_cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names()
    col_names = np.concatenate([COLLIST_NUMERIC, enc_cat_features])
    print("\nafter pp : FEATURE data shape : ", pp_data.shape)
    print("(IMP) column ({} columns): {}".format(len(col_names), col_names))
    print("sample data : \n", pp_data[0])
    
    # step_5 : persist the model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(preprocessor, path)
    print("\nsaved model at : ", str(path))
    
    
def input_fn(input_data, content_type):
    """
    std func : provide input df to the pre-processor <*ONLY during transform/predict>
        i.e load data from S3, add column names to it
       
    if col_to_predict present in input_data:
        appned it to and pass on:
    else:
        process without col_to_predict
        
    Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    print("{} input_fn {}".format("="*40, "="*40))
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        
        if len(df.columns) == len(COLLIST_ALL):
            print("training data i.e includes the col_to_predict")
            print("df.shape : ", df.shape)
            df.columns = COLLIST_ALL
        elif len(df.columns) == len(COLLIST_FEATURE):
            print("This pred/val data i.e unlabelled, not include col_to_predict")
            print("df.shape : ", df.shape)
            df.columns = COLLIST_FEATURE
        else:
            print("num cols : {}".format(len(df)))
            raise ValueError("cols assignment not handled properly")
            
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        


def predict_fn(input_data, model):
    """
    imppp : input_data : comming from input_fn
    
    std func to execute preprocessor.transform(), fit() funcs.
    
    Preprocess input data
    
    if col_to_predict present in input_data:
        appned it to and pass on:
    else:
        process without col_to_predict
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    print("{} predict_fn {}".format("="*40, "="*40))
    
    # it anyhow filter out the label_column columns (as per the script)
    print("before pp :  data shape : {}".format(input_data.shape))
    print("input data type : {}".format(type(input_data)))
    #print("input data : {}".format(input_data))
    pp_data = model.transform(input_data[COLLIST_FEATURE]) # pp_data is numpy array
    
    if col_to_predict in input_data:
        # this section used for training (NOT pred/pipeline)
        pp_data = pd.DataFrame(pp_data)
        pp_data.insert(0, col_to_predict, input_data[col_to_predict])
        pp_data.insert(1, col_primary_identifer, input_data[col_primary_identifer])
        pp_data = pp_data.values
        print("training job")
        print("below data includes : index_0: col_to_predict, index_1: col_primary_identifer")
        print("after pp : data shape : {}".format(pp_data.shape))
        print("sample data : \n", pp_data[0])
        return pp_data
    else:
        # this section used for pred/pipeline (Imp), here pass the input data without col_to_predict
        print("test/pred job\nonly contain the feature data")
        print("after pp : data shape : {}".format(pp_data.shape))
        print("sample data : \n", pp_data[0])
        return pp_data
    
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
