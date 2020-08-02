import pymongo
import json
import pandas as pd
import csv
# from logger.logconfig import getlogger

class DatabaseConnect():
	def __init__(self):
		try:
			client = pymongo.MongoClient("mongodb+srv://admin:admin@breastcancerdetection.h1ao7.mongodb.net/breast_cancer_detection?retryWrites=true&w=majority")
			self.db = client['breast_cancer_detection']

		except Exception as e:
			self.logger.info(e)
			raise e 

	def storeTrainTestCSVToDB(self, df):
		collection = self.db['corrected_train_data']
		collection.remove()
		# data_dict = df.to_dict("records")
		# collection.insert_many(data_dict)

	def storePredictCSVToDB(self, df):
		collection = self.db['predict_data']
		collection.remove()
		# data_dict = df.to_dict("records")
		# collection.insert_many(data_dict)

	def storePredictedResult(self, each_pred_record):
		
		collection = self.db['predicted_result']
		# collection.remove()
		# collection.insert_one(each_pred_record)
 	
	def fetchPredictedResults(self):
		collection = self.db['predicted_result']
		cursor = collection.find({})
		return cursor
