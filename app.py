from flask import Flask, jsonify,request, Response, render_template, redirect, session
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
import time
import atexit
from prediction_scheduler import PredictScheduler
import config
import bcrypt
import flask_monitoringdashboard as dashboard

# Flask app initialization
app = Flask(__name__)
CORS(app, supports_credentials=True)
dashboard.bind(app)

# Initilization the object of ConfigParser 
configs = configparser.ConfigParser()
configs.read('./config/bcconfig.ini')

prediction_logger = configs['logger_files']['prediction_logger']
training_logger = configs['logger_files']['training_logger']
best_pickle_file_path = configs['models']['best_pickle_file_path']


# Logger file for prediction
logger = getlogger(app.name, './logger/' + prediction_logger, consoleHandlerrequired=True)
db_conn = DatabaseConnect()

# function to render home page in GET method and predict individual values which user enter from front end or send as api using postman/insomania in POST method
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # db_conn = DatabaseConnect()
        # db_conn.fetchPredictedResults()
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
                    return render_template("index.html", pred_success_message=True)
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
    try:
        result = db_conn.fetchPredictedResults()
        return render_template("predicted-results.html", predicted_result=result)

    except Exception as e:
        logger.info(e)
        raise e


# function to download predicted result from database
@app.route("/predicted-results-download", methods=['GET'])
def downloadPredictedCsv():
    cursor = db_conn.fetchPredictedResults()
    
    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))
    curr_time = time.localtime() 
    curr_clock = time.strftime("%d-%b-%Y %H:%M", curr_time)
    return Response(df.to_csv(), mimetype="text/csv", headers={"Content-disposition":
       "attachment; filename=predicted_result_"+ curr_clock + ".csv"})


################### Routes for Admin Dashboard ###################
# function to check login for custom dashboard
@app.route('/admin', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        # db_conn = DatabaseConnect()
        # db_conn.addUser()
        if 'email' in session:
            return redirect('admin/dashboard')

        else:
            return render_template('dashboard/login.html')

    elif request.method == 'POST':
        login_user = db_conn.userLogin(request.form['email'])
        # print(login_user)
        # print(login_user['password'].encode('utf-8'))
        if login_user:
            if request.form['password'] == login_user['password']:
                session['email'] = request.form['email']
                return redirect('admin/dashboard')
            else:
                return render_template('dashboard/login.html', login_error=True)
        else:
            return render_template('dashboard/login.html', login_error=True)

# custom dashboard logout function
@app.route('/admin/logout', methods=['GET'])
def logout():
    if 'email' in session:
        session.clear()
        return redirect('/admin')

def generate(type):
    if type == "training":
        with open('logger/training_logs.log') as f:
            yield f.read()
    elif type == "Prediction":
        with open('logger/prediction_logs.log') as f:
            yield f.read()

# function to generate reporting on custom dashboard
@app.route("/admin/reporting", methods=['GET'])
def reporting():
    if request.method == 'GET':
        type = request.args.get('type')
        return app.response_class(generate(type), mimetype='text/plain')
 
# function to display details on home custom Dashboard home page
@app.route("/admin/dashboard", methods=['GET'])
def dashboard():
    if 'email' in session:
        # function call to display dashboard details from DB
        result = db_conn.fetchDashboardDetails()
        split_result = result['dsbrd_target_count'].split("/")
        # function call to getch data for charts
        return render_template('dashboard_base.html', dashboard_details=result, malignant = split_result[0], benign = split_result[1])
    else:
        return redirect('/admin')   
if __name__ == '__main__':
    app.secret_key = 'asfasdasf#$@!$@!#asfadasdasd!@$!#'
    app.run(port=config.PORT, debug=config.DEBUG_MODE)