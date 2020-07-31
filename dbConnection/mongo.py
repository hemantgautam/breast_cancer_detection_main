import pymongo
import json
import pandas as pd
import csv
# from logger.logconfig import getlogger

class DatabaseConnect():
	def __init__(self):
		try:
			# self.logger = getlogger(
			# __name__, './logger/training_logs.log', consoleHandlerrequired=True)
			client = pymongo.MongoClient("mongodb+srv://admin:admin@breastcancerdetection.h1ao7.mongodb.net/breast_cancer_detection?retryWrites=true&w=majority")
			self.db = client['breast_cancer_detection']
			# collection = db['corrected_train_data']
			# collection.remove({})
			# collection.removeMany({})
			# user_dict = {"_id": 200, "name": "Gautam"}
			# collection.insert_one(user_dict)
			# print(collection)
		except Exception as e:
			# self.logger.info(e)
			# self.logger.info("Error in Connection with Database")
			raise e 

	def storeTrainTestCSVToDB(self, df):
		collection = self.db['corrected_train_data']
		# try:
		# 	csvfile = open('train_test_data/final_data_for_model/final_data.csv', 'r')
		# 	reader = csv.DictReader( csvfile )
		# 	header= list(df.columns)
		# 	collection.remove()
		# 	for each in reader:
		# 	    row={}
		# 	    for field in header:
		# 	        row[field]=each[field]

		# 	    # collection.insert_one(row)
		# except Exception as e:
		# 	# self.logger.info(e)
		# 	raise e
		collection.remove()
		data_dict = df.to_dict("records")
		# Insert collection
		collection.insert_many(data_dict)

	# def saveBestModelPklName(self, pkl_file_name):
	# 	collection = self.db['best_model_pkl_file_name']
	# 	collection.remove()
	# 	pkl_dict = {"model_pkl_name": pkl_file_name}
	# 	collection.insert_one(pkl_dict)


	# def fetchBestModelPklName(self):
	# 	collection = self.db['best_model_pkl_file_name']
	# 	return collection.find()

	def storePredictCSVToDB(self, df):
		collection = self.db['predict_data']
		collection.remove()
		data_dict = df.to_dict("records")
		collection.insert_many(data_dict)


	def storePredictedResult(self, result_dict):
		
		collection = self.db['predicted_result']
		# collection.remove()
		collection.insert_one(result_dict)
 