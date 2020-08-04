import pymongo
import json
import pandas as pd
import csv

# Class for all CURD operations on MongoDB
class DatabaseConnect():
    def __init__(self):

        # Exception handling if connection with mongodb fails
        try:

            # Connection string for MongoDB
            client = pymongo.MongoClient(
                "mongodb+srv://admin:admin@breastcancerdetection.h1ao7.mongodb.net/breast_cancer_detection?retryWrites=true&w=majority")

            # "breast_cancer_detection" is the Database Name
            self.db = client['breast_cancer_detection']

        except Exception as e:
            self.logger.info(e)
            raise e

    # function to store train/test data into Database
    def storeTrainTestCSVToDB(self, df):
        collection = self.db['corrected_train_data']
        collection.remove()
        data_dict = df.to_dict("records")
        collection.insert_many(data_dict)

    # function to store prediction csv into Database
    def storePredictCSVToDB(self, df):
        collection = self.db['predict_data']
        collection.remove()
        data_dict = df.to_dict("records")
        collection.insert_many(data_dict)

    # function to store predicted results(id and cancer type) into Database
    def storePredictedResult(self, df):
        collection = self.db['predicted_result']
        collection.remove()
        data_dict = df.to_dict("records")
        collection.insert_many(data_dict)

    # function to fetch predicted results(id and cancer type) and show in front end
    def fetchPredictedResults(self):
        collection = self.db['predicted_result']
        # collection.remove()
        cursor = collection.find({})
        return cursor
