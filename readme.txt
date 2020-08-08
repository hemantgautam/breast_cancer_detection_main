Created by -
	alam.jane61@gmail.com and hemantgautam50@gmail.com

Project Description:
	The main purpose of this project is to predict the cancer type of given data.
	Their are two type of cancer given in training Data - 
	1. malignant: represents with 1
	2. benign: represents with 0

	First models are getting trained on train and test data and best model is chosen based on the accuracy and that model is used for prediction.


Features - 
1. Training the model with api(/breast-cancer-api/train) using postman/insomania.
		a. Training will create 4 pickle files inside models/ directory
		b. based on the accuracy best pickle file will be renamed as "best_pickle_file.pkl" and moved inside models/final_model directory

2. Predicting the values in three ways - 
		a. Single record prediction from UI(single_record_predition.py)
		b. Single record prediction from Json
		c. Csv upload data prediction

3. Logger
	a. Their are two logger files getting created for training and prediction seperately.
	b. All train/predict methods execution will be captured in respective logger files.
	c. All exceptions will also be captured in logger file.

4. Train/test csv data(breast_cancer_dataset.csv) is kept inside "train_test_data" directory, this is the file which will be used to train models.

5. After all imputation and cleaning processes of the train csv, final cleaned csv is getting stored inside "train_test_data/final_data_for_model" directory.

6. If csv is getting uploaded for prediction from UI, that csv is getting stored inside" predict_csv_uploads" directory.

7. All clean train, predict csv and predicted results stores in Mongo DB.

8. Documentation directory contains details LLD document for the project.

9. EDA directory contains all ipynb files and their screenshot.

10. All html files are kept inside templates directory.

11. Model is deployed on Azure and to access that url is:
	http://breastcancermodeldetection.azurewebsites.net/





