import os
import sys
import pandas as pd
from src.CreditCardFaultDetection.exception import customexception
from src.CreditCardFaultDetection.logger import logging
from src.CreditCardFaultDetection.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            raise customexception(e,sys)
    
    
class CustomData:
    def __init__(self,
                 online_order:object,
                 book_table:object,
                 votes:int,
                 location:object,
                 rest_type:object,
                 cuisines:object,
                 cost_for_2:object,
                 type:object
                 ):
        
        #online_order book_table location rest_type cuisines cost_for_2 type votes
        
        self.online_order=online_order
        self.book_table=book_table
        self.votes=votes
        self.location=location
        self.rest_type=rest_type
        self.cuisines=cuisines
        self.cost_for_2=cost_for_2
        self.type=type
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'online_order':[self.online_order], 
                    'book_table':[self.book_table], 
                    'votes':[self.votes], 
                    'location':[self.location], 
                    'rest_type':[self.rest_type], 
                    'cuisines':[self.cuisines],
                    'cost_for_2':[self.cost_for_2],
                    'type':[self.type]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)