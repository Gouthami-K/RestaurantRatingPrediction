import pandas as pd
import numpy as np
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import customexception
from src.RestaurantRatingPrediction.data_access.mongo_conn import MongoConnector


import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
            # to retrieve data from MongoDB
            mongo_uri = "mongodb+srv://zomato:zomato@zomato.0zgtc3p.mongodb.net/?retryWrites=true&w=majority"
            database_name = "zomato_database"
            collection_name = "rest_reviews"

            mongodb_data = MongoConnector.get_data_from_database(mongo_uri, database_name, collection_name)
    
            data = pd.DataFrame(mongodb_data)
            logging.info(" I have read dataset from MongoDB as a df")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.33,random_state=10)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)
    