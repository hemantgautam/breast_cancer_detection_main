from flask import Flask, jsonify,request, Response, render_template, redirect
from flask_cors import CORS, cross_origin
import pickle
import sklearn
import csv
import pandas as pd
import os
from logger.logconfig import getlogger
from train_validation_process.train_validation import TrainValidation
from predict_validation_process.predict_validation import PredictValidation
from dbConnection.mongo import DatabaseConnect
from single_record_prediction import SingleRecordPrediction
import configparser


# Flask app initialization
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Initilization the object of ConfigParser 
config = configparser.ConfigParser()
config.read('./config/bcconfig.ini')

prediction_logger = config['logger_files']['prediction_logger']
training_logger = config['logger_files']['training_logger']
best_pickle_file_path = config['models']['best_pickle_file_path']


# Logger file for prediction
logger = getlogger(app.name, './logger/' + prediction_logger, consoleHandlerrequired=True)


# function to render home page in GET method and predict individual values which user enter from front end or send as api using postman/insomania in POST method
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        return SingleRecordPrediction("form", request.form)


# API for training models using postman/insomania software
@app.route("/breast-cancer-api/train", methods=['GET'])
def trainValidationAPI():
    trainObject = TrainValidation()
    response = trainObject.train_validation()
    if response is True:
        return jsonify({'message': 'Model training successful. Start predicting from web application.'})
    else:
        return jsonify({'message': 'Model training fails. Please check the training logs.'})

@app.route("/predict-api", methods=['POST'])
def predict_api():
    return SingleRecordPrediction("json", request.json)
    


# function to predict csv file data and store result into database
@app.route("/predict", methods=['GET', 'POST'])
def predictValidation():
    isfilepresent = os.path.isfile(best_pickle_file_path)
    if isfilepresent:
        if request.method == 'POST':
            f = request.files['file']
            filename = f.filename
            file_type = filename.split(".")[-1]
            if file_type.lower() == 'csv':
                logger.info(
                    "==========================Prediction  Started==========================")
                logger.info(f.filename)
                f.save("predict_csv_uploads/" + filename)
                df = pd.read_csv("predict_csv_uploads/" + filename, encoding='ISO-8859-1')
                predObject = PredictValidation(df)
                response = predObject.predict_validation()
                if response is True:
                    return render_template("index.html", success_message="Prediction successfully completed. Please check result in 'Predicted Results' link.")
                else:
                    return render_template("index.html", error_message="Data Prediction Failed. Please check the predicton logs.")
            else:
                return render_template("index.html", error_message="File type should be CSV")
        else:
            return redirect("/")
    else:
        return render_template("index.html", error_message="No model present. Please train the model first.")


# function to show all csv predicted data in tabuler data
@app.route("/predicted-results", methods=['GET'])
def predictedResult():
    db_conn = DatabaseConnect()
    try:
        result = db_conn.fetchPredictedResults()
        return render_template("predicted-results.html", predicted_result=result)

    except Exception as e:
        logger.info(e)
        raise e

 
if __name__ == '__main__':
    app.run(debug=True)
