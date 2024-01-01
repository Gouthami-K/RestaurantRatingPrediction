from src.RestaurantRatingPrediction.logger import logging
from src.RestaurantRatingPrediction.exception import customexception
import pandas as pd
from pymongo.mongo_client import MongoClient

class MongoConnector:
    @staticmethod
    def connect_database(uri):
        """
        The connect_database function establishes a connection to the MongoDB database.
        
        :param uri: str: URI of mongodb atlas database
        :return: MongoClient: A mongoclient object
        """       
        client = MongoClient(uri)
        try:
            client.admin.command('ping')
            logging.info("MongoDb connection established successfully")
            return client
        except Exception as e:
            logging.error("Exception occurred during creating database connection")
            raise customexception(e, sys)

    @staticmethod
    def get_data_from_database(uri, database_name, collection_name):
        """
        The get_data_from_database function takes in a URI, database name, and collection name,
        connects to the database, and returns a pandas dataframe of the data from that collection.

        :param uri: str: MongoDB database URI
        :param database_name: str: Database name
        :param collection_name: str: Collection name
        :return: DataFrame: A pandas dataframe
        """
        client = MongoConnector.connect_database(uri)
        collection = client[database_name][collection_name]
        data = list(collection.find())
        client.close()  # Close the MongoDB connection
        return pd.DataFrame(data)