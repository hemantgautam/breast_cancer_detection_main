from logger.logconfig import getlogger
from data_processing_functions import DataProcessingFunctions

# Class to call all Prediction methods under "predict_validation" function
class PredictValidation:

    def __init__(self, df):

        # initializing logger file for prediction
        self.logger = getlogger(
            __name__, './logger/prediction_logs.log', consoleHandlerrequired=True)

        # Creating object of DataProcessingFunctions class
        self.data_process_funs = DataProcessingFunctions("prediction_logs.log", "prediction", df)

    # Main function which calls each prediction function one by one
    def predict_validation(self):

        # validating the response of matchColumnsDetailsWithSchema function, other function calls depends on the response on this function.
        response = self.data_process_funs.matchColumnsDetailsWithSchema()
        if response is True:

            # function to remove null values from dataset
            self.data_process_funs.removeNullValues()

            # function to predict values
            return self.data_process_funs.predictValues()

        else:
            self.logger.info("matchColumnsDetailsWithSchema Failed")
            return False
