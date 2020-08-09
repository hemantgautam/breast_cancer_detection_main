from flask import jsonify, render_template
import os
import pickle
from logger.logconfig import getlogger

# Logger file for prediction
logger = getlogger(__name__, './logger/prediction_logs.log', consoleHandlerrequired=True)
best_pkl_file_path = "models/final_model/best_pickle_file.pkl"

# function to predict single record
def SingleRecordPrediction(request_type, request_body):

    # check is model is trained or not, if its not trained, best pickle file wont be present under models/final_model
    isfilepresent = os.path.isfile(best_pkl_file_path)
    if isfilepresent:
        
        # if the request is in a json format coming from postman/insomania
        if  request_body is not None:
            try:
                radius_mean = float(request_body['radius_mean'])
                texture_mean = float(request_body['texture_mean'])
                perimeter_mean = float(request_body['perimeter_mean'])
                area_mean = float(request_body['area_mean'])
                smoothness_mean = float(request_body['smoothness_mean'])
                compactness_mean = float(request_body['compactness_mean'])
                concavity_mean = float(request_body['concavity_mean'])
                concave_points_mean = float(request_body['concave_points_mean'])
                symmetry_mean = float(request_body['symmetry_mean'])
                fractal_dimension_mean = float(request_body['fractal_dimension_mean'])
                radius_se = float(request_body['radius_se'])
                texture_se = float(request_body['texture_se'])
                perimeter_se = float(request_body['perimeter_se'])
                area_se = float(request_body['area_se'])
                smoothness_se = float(request_body['smoothness_se'])
                compactness_se = float(request_body['compactness_se'])
                concavity_se = float(request_body['concavity_se'])
                concave_points_se = float(request_body['concave_points_se'])
                symmetry_se = float(request_body['symmetry_se'])
                fractal_dimension_se = float(request_body['fractal_dimension_se'])
                radius_worst = float(request_body['radius_worst'])
                texture_worst = float(request_body['texture_worst'])
                perimeter_worst = float(request_body['perimeter_worst'])
                area_worst = float(request_body['area_worst'])
                smoothness_worst = float(request_body['smoothness_worst'])
                compactness_worst = float(request_body['compactness_worst'])
                concavity_worst = float(request_body['concavity_worst'])
                concave_points_worst = float(request_body['concave_points_worst'])
                symmetry_worst = float(request_body['symmetry_worst'])
                fractal_dimension_worst = float(request_body['fractal_dimension_worst'])
            
            except Exception as e:

                # returning exception based on the request type(json/form)
                if request_type == "json":
                    res = {"Error": e}
                    logger.info(e)
                    return jsonify(res)
                else:
                    return render_template('index.html', error_text=e)
        try:

            # loading the best pickle file to predict data
            model = pickle.load(
                open(best_pkl_file_path, 'rb'))

            # predicting the data
            predicted_value = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])
            
            if predicted_value[0] == 1:
                prediction_text = "You Cancer type is Malignant"
            elif predicted_value[0] == 0:
                prediction_text = "You Cancer type is Benign"
            else:
                prediction_text = "No prediction found!"
            
            # returning cancer type based on the request type(json/form)    
            if request_type == "json":
                res = {"Cancer Type": prediction_text}
                logger.info(res)
                return jsonify(res)
            else:
                logger.info(prediction_text)
                return render_template('index.html', prediction_text="{}".format(prediction_text))
        
        except Exception as e:

            # returning exception based on the request type(json/form)
            if request_type == "json":
                res = {"Error": e}
                logger.info(e)
                return jsonify(res)
            else:            
                return render_template('index.html', error_text=e)
    
    else:
        if request_type == "json":
            res = {"Error": "Model file is not readable/present"}
            logger.info(res)
            return jsonify(res)
        else:
            return render_template("index.html", file_error=True)
