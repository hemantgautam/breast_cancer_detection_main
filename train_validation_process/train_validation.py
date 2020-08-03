from logger.logconfig import getlogger
from train_validation_process.train_validation_functions import TrainValidationFunctions

# Class to call all Training methods under "train_validation" function
class TrainValidation:

    def __init__(self):

        # initializing logger file for training
        self.logger = getlogger(
            __name__, './logger/training_logs.log', consoleHandlerrequired=True)
        self.logger.info(
            "==========================Training  Started==========================")

        # Creating object of TrainValidationFunctions class, which has all the training related functions
        self.train_val_funs = TrainValidationFunctions()

    # Main function which calls each training function one by one
    def train_validation(self):

        # validating the response of matchColumnsDetailsWithSchema function, other function calls depends on the response on this function.
        response = self.train_val_funs.matchColumnsDetailsWithSchema()
        if response is True:

            # function to remove null values from dataset
            self.train_val_funs.removeNullValues()

            # function to convert cateorical data into numbers
            self.train_val_funs.convertToDummies()

            # function to store clean data under "train_test_data/final_data_for_models" folder in csv format
            self.train_val_funs.createFinalDataForTrainingModels()

            # function to save clean csv into database also
            self.train_val_funs.storeFinalCsvToDatabase()

            # Most important function which test the different models and generate pickle files under "model" folder
            return self.train_val_funs.ModelSelection()
        else:
            self.logger.info("matchColumnsDetailsWithSchema Failed")
            return False
