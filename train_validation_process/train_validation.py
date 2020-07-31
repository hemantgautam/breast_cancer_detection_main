import pandas as pd
from logger.logconfig import getlogger
import os
from train_validation_process.train_validation_functions import TrainValidationFunctions

class TrainValidation:
	
	def __init__(self):
		self.logger = getlogger(__name__, './logger/training_logs.log', consoleHandlerrequired=True)
		self.logger.info("==========================Training  Started==========================")
		self.train_val_funs = TrainValidationFunctions()		
	
	def train_validation(self):
		response = self.train_val_funs.matchColumnsDetailsWithSchema()
		if response is True:

			# function remove null Values in axis 0
			self.train_val_funs.removeNullValues()

			self.train_val_funs.convertToDummies()
			
			self.train_val_funs.createFinalDataForTrainingModels()

			self.train_val_funs.storeFinalCsvToDatabase()
			
			return self.train_val_funs.ModelSelection()
			# self.train_val_funs.selectModel()
			# function to store data into Mongo db
			# read data from DB and store that data into another folder to create model
			# function for creating model


			# self.logger.info("===========Train Successsful completed.===========")
		else:
			self.logger.info("matchColumnsDetailsWithSchema Failed")
			return False