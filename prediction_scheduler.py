from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
from predict_validation_process.predict_validation import PredictValidation
import configparser
from logger.logconfig import getlogger

# Initilization the object of ConfigParser 
config = configparser.ConfigParser()
config.read('./config/bcconfig.ini')

prediction_logger = config['logger_files']['prediction_logger']

# Logger file for prediction
logger = getlogger(__name__, './logger/' + prediction_logger, consoleHandlerrequired=True)

sched = BackgroundScheduler()


'''
    function to run schedule job to predict continues data,
    as of now we have only one csv file, and the same file 
    we are using for train, test and predict also. 
    This is getting called using decorator and will be triggered every 8 hours, prdict the data and will store into DB.
''' 
@sched.scheduled_job(trigger="interval", hours=8)
def PredictScheduler():
    logger.info("================Prediction Scheduler Started================")
    try:
        df = pd.read_csv("predict_csv_uploads/Breast Cancer-Test-Data1.csv")
        predObject = PredictValidation(df)
        response = predObject.predict_validation()
        if response is True:
            logger.info("Prediction Successful")
        else:
            logger.info("Prediction Failed")
    except:
        logger.info("Error in reading file.")    
    logger.info("================Prediction Scheduler Ended================")

sched.start()