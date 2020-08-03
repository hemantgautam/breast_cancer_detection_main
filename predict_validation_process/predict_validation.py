from logger.logconfig import getlogger
from predict_validation_process.predict_validation_functions import PredictValidationFunctions

# Class to call all Prediction methods under "predict_validation" function
class PredictValidation:

    def __init__(self, df):

        # initializing logger file for prediction
        self.logger = getlogger(
            __name__, './logger/prediction_logs.log', consoleHandlerrequired=True)

        # Creating object of PredictValidationFunctions class, which has all the prediction related functions
        self.predict_val_funs = PredictValidationFunctions(df)

    # Main function which calls each prediction function one by one
    def predict_validation(self):

        # validating the response of matchColumnsDetailsWithSchema function, other function calls depends on the response on this function.
        response = self.predict_val_funs.matchColumnsDetailsWithSchema()
        if response is True:

            # function to remove null values from dataset
            self.predict_val_funs.removeNullValues()

            # function to predict values
            return self.predict_val_funs.predictValues()
            # self.train_val_funs.selectModel()
            # function to store data into Mongo db
            # read data from DB and store that data into another folder to create model
            # function for creating model

            # self.logger.info("===========Train Successsful completed.===========")
        else:
            self.logger.info("matchColumnsDetailsWithSchema Failed")
            return False
