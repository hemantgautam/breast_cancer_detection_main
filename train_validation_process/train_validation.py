from logger.logconfig import getlogger
from data_processing_functions import DataProcessingFunctions
import configparser

# Initilization the object of ConfigParser 
config = configparser.ConfigParser()
config.read('./config/bcconfig.ini')

training_logger = config['logger_files']['training_logger']

# Class to call all Training methods under "train_validation" function
class TrainValidation:

    def __init__(self):

        # initializing logger file for training
        self.logger = getlogger(
            __name__, './logger/' + training_logger, consoleHandlerrequired=True)
        self.logger.info(
            "==========================Training  Started==========================")

        # Creating object of DataProcessingFunctions class
        self.data_process_funs = DataProcessingFunctions(training_logger, "training")

    # Main function which calls each training function one by one
    def train_validation(self):

        # validating the response of matchColumnsDetailsWithSchema function, other function calls depends on the response on this function.
        response = self.data_process_funs.matchColumnsDetailsWithSchema()
        if response is True:

            # function to remove null values from dataset
            self.data_process_funs.removeNullValues()

            # function to convert cateorical data into numbers
            self.data_process_funs.convertToDummies()

            # function to store clean data under "train_test_data/final_data_for_models" folder in csv format
            self.data_process_funs.createFinalDataForTrainingModels()

            # function to save clean csv into database also
            self.data_process_funs.storeFinalCsvToDatabase()

            # Most important function which test the different models and generate pickle files under "model" folder
            return self.data_process_funs.ModelSelection()
        else:
            self.logger.info("matchColumnsDetailsWithSchema Failed")
            return False
