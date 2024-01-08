import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import customexception

from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            
            # Train model on the full training set (no need to fit it twice)
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Get R2 scores for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {'train_score': train_model_score, 'test_score': test_model_score}

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')

        raise customexception(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)
    
import numpy as np


def handle_categorical_columns(data, column_thresholds):
    try:
        for column_name, threshold in column_thresholds.items():
            column_count = data[column_name].value_counts()
            categories_below_threshold = column_count[column_count < threshold].index
            data[column_name] = np.where(data[column_name].isin(categories_below_threshold), 'others', data[column_name])
            print(f"Updated column '{column_name}' with threshold {threshold}")

        #return data
        
    except Exception as e:
        logging.error(f"Error in handling categorical columns: {e}")
        raise customexception(e, sys)


def handle_rate_column(df, column_name="rate"):
    try:
        df[column_name] = df[column_name].apply(lambda value: np.nan if value in ["NEW", "-"] else float(str(value).split("/")[0]))
        
        # Replacing null values with the mean
        df[column_name].fillna(df[column_name].mean(), inplace=True)

        #return df
        
    except Exception as e:
        logging.error(f"Error in handling {column_name} column: {e}")
        raise customexception(e, sys)

     
     

