from src.RestaurantRatingPrediction.components.data_ingestion import DataIngestion

import os
import sys
from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import customexception
import pandas as pd

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()