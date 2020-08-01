from flask import Flask, jsonify, Response, render_template, redirect
from flask import request, json, escape
from flask_cors import CORS, cross_origin
from wsgiref import simple_server
import pickle
import sklearn
import csv
import pandas as pd
import os
from logger.logconfig import getlogger
from train_validation_process.train_validation import TrainValidation
from predict_validation_process.predict_validation import PredictValidation
from dbConnection.mongo import DatabaseConnect
import pymongo

app = Flask(__name__)
CORS(app, supports_credentials=True)
logger = getlogger(app.name, './logger/prediction_logs.log', consoleHandlerrequired=True)


@app.route("/apibreastcancer", methods=["GET"])
# @cross_origin(supports_credentials=True)
def index_page():
    response = jsonify('Hello World!!!')
    return response

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/train", methods=['GET'])
def trainValidation():
    trainObject = TrainValidation()
    response = trainObject.train_validation()
    if response is True:
        return render_template('index.html', model_training_success=True)
    else:
        return render_template('index.html', model_training_failure=False)

@app.route("/predict", methods=['GET', 'POST'])
def predictValidation():
    isfilepresent = os.path.isfile('model/final_model/best_pickle_file.pkl')
    if isfilepresent:
        if request.method == 'POST':
            f = request.files['file']
            filename = f.filename
            file_type = filename.split(".")[-1]
            if file_type.lower() == 'csv':
                logger.info("==========================Prediction  Started==========================")
                logger.info(f.filename)
                f.save("uploads/" + filename)
                df = pd.read_csv(f.filename, encoding='ISO-8859-1')
                # X = df.drop(columns=['id'], axis=1)
                predObject = PredictValidation(df)
                response = predObject.predict_validation()
                return render_template("index.html", predict_result=response)
        else:
            return redirect("/")
    else:
        return render_template("index.html", file_error=True)


@app.route("/predicted-results", methods=['GET'])
def predictedResult():
    db_conn = DatabaseConnect()
    result = db_conn.fetchPredictedResults()
    return render_template("predicted-results.html", predicted_result=result)

@app.route("/", methods=['POST'])
def predict():
    isfilepresent = os.path.isfile('model/final_model/best_pickle_file.pkl')
    if isfilepresent:
        try:
            radius_mean = float(request.form['radius_mean'])
            texture_mean = float(request.form['texture_mean'])
            perimeter_mean = float(request.form['perimeter_mean'])
            area_mean = float(request.form['area_mean'])
            smoothness_mean = float(request.form['smoothness_mean'])
            compactness_mean = float(request.form['compactness_mean'])
            concavity_mean = float(request.form['concavity_mean'])
            concave_points_mean = float(request.form['concave_points_mean'])
            symmetry_mean = float(request.form['symmetry_mean'])
            fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
            radius_se = float(request.form['radius_se'])
            texture_se = float(request.form['texture_se'])
            perimeter_se = float(request.form['perimeter_se'])
            area_se = float(request.form['area_se'])
            smoothness_se = float(request.form['smoothness_se'])
            compactness_se = float(request.form['compactness_se'])
            concavity_se = float(request.form['concavity_se'])
            concave_points_se = float(request.form['concave_points_se'])
            symmetry_se = float(request.form['symmetry_se'])
            fractal_dimension_se = float(request.form['fractal_dimension_se'])
            radius_worst = float(request.form['radius_worst'])
            texture_worst = float(request.form['texture_worst'])
            perimeter_worst = float(request.form['perimeter_worst'])
            area_worst = float(request.form['area_worst'])
            smoothness_worst = float(request.form['smoothness_worst'])
            compactness_worst = float(request.form['compactness_worst'])
            concavity_worst = float(request.form['concavity_worst'])
            concave_points_worst = float(request.form['concave_points_worst'])
            symmetry_worst = float(request.form['symmetry_worst'])
            fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
            try:
                model = pickle.load(open("model/final_model/best_pickle_file.pkl", 'rb'))
                predicted_value = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])
                if predicted_value[0] == 1:
                    prediction_text = "You Cancer type is Malignant"
                elif predicted_value[0] == 0:
                    prediction_text = "You Cancer type is Benign"
                else:
                    prediction_text = "No prediction found!"
                return render_template('index.html', prediction_text="{}".format(prediction_text))
            except Exception as e:
                return render_template('index.html', error_text=e) 

        except Exception as e:
            # logger.info(e)
            return render_template('index.html', error_text=e) 
    else:
        return render_template("index.html", file_error=True)


if __name__ == '__main__':
    app.run(port='6000')
    # app.run(debug=True)
