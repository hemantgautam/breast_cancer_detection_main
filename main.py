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
from single_record_predition import SingleRecordPrediction

# Flask app initialization
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Logger file for prediction
logger = getlogger(app.name, './logger/prediction_logs.log', consoleHandlerrequired=True)


# function to render home page in GET method and predict individual values which user enter from front end or send as api using postman/insomania in POST method
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        if request.json is not None:
            return SingleRecordPrediction("json", request.json)
        else:
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


# function to predict csv file data and store result into database
@app.route("/predict", methods=['GET', 'POST'])
def predictValidation():
    isfilepresent = os.path.isfile('models/final_model/best_pickle_file.pkl')
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
                return render_template("index.html", predict_result=response)
        else:
            return redirect("/")
    else:
        return render_template("index.html", file_error=True)


# function to show all csv predicted data in tabuler data
@app.route("/predicted-results", methods=['GET'])
def predictedResult():
    db_conn = DatabaseConnect()
    try:
        result = db_conn.fetchPredictedResults()
        return render_template("predicted-results.html", predicted_result=result)

    except Exception as e:
        self.logger.info(e)
        raise e


if __name__ == '__main__':
    app.run(debug=True)
