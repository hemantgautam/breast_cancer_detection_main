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

			self.train_val_funs.removeNullValues()

			self.train_val_funs.convertToDummies()
			
			self.train_val_funs.createFinalDataForTrainingModels()

			self.train_val_funs.storeFinalCsvToDatabase()
			
			return self.train_val_funs.ModelSelection()
		else:
			self.logger.info("matchColumnsDetailsWithSchema Failed")
			return False