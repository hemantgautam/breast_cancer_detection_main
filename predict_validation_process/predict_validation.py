import pandas as pd
from logger.logconfig import getlogger
import os
from predict_validation_process.predict_validation_functions import PredictValidationFunctions

class PredictValidation:
	
	def __init__(self, df):
		self.logger = getlogger(__name__, './logger/prediction_logs.log', consoleHandlerrequired=True)
		self.predict_val_funs = PredictValidationFunctions(df)		
	
	def predict_validation(self):
		response = self.predict_val_funs.matchColumnsDetailsWithSchema()
		if response is True:

			# function remove null Values in axis 0
			self.predict_val_funs.removeNullValues()

			return self.predict_val_funs.predictValues()
			# self.train_val_funs.selectModel()
			# function to store data into Mongo db
			# read data from DB and store that data into another folder to create model
			# function for creating model


			# self.logger.info("===========Train Successsful completed.===========")
		else:
			self.logger.info("matchColumnsDetailsWithSchema Failed")
			return False