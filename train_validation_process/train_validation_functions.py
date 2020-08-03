import pandas as pd
from logger.logconfig import getlogger
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from dbConnection.mongo import DatabaseConnect
from shutil import copyfile

# initilizing DB connection
db_conn = DatabaseConnect()

# class for creating all training methods
class TrainValidationFunctions:
    def __init__(self):

        # initializing logger file for training
        self.logger = getlogger(
            __name__, './logger/training_logs.log', consoleHandlerrequired=True)

        # schema json file which is getting used to validate training csv data
        self.schema_path = 'schema_training.json'

        # Reading trainings csv file to perform training
        self.df = pd.read_csv('train_test_data/breast_cancer_dataset.csv')
        self.logger.info("---Shape Before deleting unnammed column---")
        self.logger.info(self.df.shape)

        # Dropping all unnamed data from dataset
        self.df.drop(self.df.columns[self.df.columns.str.contains(
            'unnamed', case=False)], axis=1, inplace=True)
        self.logger.info("---Shape After deleting unnammed column---")
        self.logger.info(self.df.shape)

    # This is the most important function which matches the number, type and nams of csv data with schema file
    def matchColumnsDetailsWithSchema(self):
        self.logger.info(
            "----------matchColumnsDetailsWithSchema Starts----------")
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            NumOfColInSchema = dic['NumberofColumns']
            SchemaColName = dic["ColName"]
            NumOfColInDf = self.df.shape[1]
            schema_col_types = list(SchemaColName.values())
            df_col_types = []
            for i in range(len(self.df.columns)):
                df_col_types.append(self.df.dtypes[i])

            # Comparing number of columns and columns names in csv and training schema
            if NumOfColInSchema == NumOfColInDf and list(SchemaColName.keys()) == list(self.df.columns):

                # Comparing type of each columns from csv with schema
                try:
                    if len(schema_col_types) == len(df_col_types) and len(schema_col_types) == sum([1 for i, j in zip(schema_col_types, df_col_types) if i == j]):
                        self.logger.info(
                            "Number of columns are identical with equal columns in the dataset and schema.")
                        return True
                    else:
                        self.logger.info(
                            "Type of columns are not similar in the dataset and schema.")
                except Exception as e:
                    self.logger.info(e)
                    self.logger.info(
                        "column type issue in schema_training.json")
            else:
                self.logger.info(
                    "Number or Names of columns in csv are not matching with schema.")
        except Exception as e:
            self.logger.info(e)
            self.logger.info("Error in reading schema_training.json file")

    # function remove the null values from csv data
    def removeNullValues(self):
        self.logger.info("----------removeNullValues Starts----------")
        self.logger.info(self.df.shape)
        self.logger.info("Null Values Count Before :")
        self.logger.info(str(self.df.isna().sum()))

        self.df.dropna(axis=0, how="any", inplace=True)

        self.logger.info("Null Values Count After :")

        self.logger.info(str(self.df.isna().sum()))

    # function to load best_pickle file, predict data and store csv and predicted data both into database
    def convertToDummies(self):
        self.logger.info("----------convertToDummies Starts----------")
        self.logger.info("Before Converting to Dummies")
        self.logger.info(self.df['diagnosis'].head())

        self.df['diagnosis'].replace(to_replace=dict(M=1, B=0), inplace=True)

        self.logger.info("After Converting to Dummies")

        self.logger.info(self.df['diagnosis'].head())

    # function to save final clean train csv into "train_test_data/final_data_for_model/" folder
    def createFinalDataForTrainingModels(self):
        try:
            self.logger.info(
                "----------createFinalDataForTrainingModels Starts----------")
            self.df.to_csv(
                'train_test_data/final_data_for_model/final_data.csv', index=False)
        except Exception as e:
            self.logger.info(e)
            raise e

    # function to store final clean csv into database
    def storeFinalCsvToDatabase(self):
        self.logger.info("----------storeFinalCsvToDatabase Starts----------")
        try:
            db_conn.storeTrainTestCSVToDB(self.df)

        except Exception as e:
            self.logger.info(e)
            self.logger.info("Error in storing DF in database")
            raise e

    # function to create model from different algorithms and save best model's pickle file under "model/final_model/" folder which will be used for prediction
    def ModelSelection(self):
        self.logger.info("----------selectModel Starts----------")
        X = self.df.drop(columns=['id', 'diagnosis'], axis=1)
        y = self.df['diagnosis']

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=5)

        # Creating models dict to run in a loop
        models = {"lr": LogisticRegression(), "rfc": RandomForestClassifier(
        ), "svc": SVC(), "knn": KNeighborsClassifier(n_neighbors=1)}

        model_acccuray = {}
        try:
            for key, value in models.items():
                dict_key = key
                key = value
                key.fit(X_train, y_train)
                pred = key.predict(X_test)
                accuray = accuracy_score(y_test, pred)
                model_acccuray[dict_key] = round(accuray*100, 2)
                filename = 'model/' + dict_key + '_breast_cancer_model_' + \
                    str(round(accuray*100, 2)) + '.pkl'
                pickle.dump(key, open(filename, 'wb'))
                self.logger.info(classification_report(y_test, pred))
                self.logger.info(confusion_matrix(y_test, pred))
            self.logger.info("Model Accuray")
            self.logger.info(model_acccuray)

            # creating best_pickle_file.pkl file from all the created pickle files
            best_model_pkl_file = sorted(model_acccuray.items(), key=lambda x: x[1], reverse=True)[
                0][0] + '_breast_cancer_model_' + str(sorted(model_acccuray.items(), key=lambda x: x[1], reverse=True)[0][1]) + '.pkl'

            # cpoying the best pickle file from under model folder to model/final_model folder which will be used for prediction
            copyfile("model/"+best_model_pkl_file,
                     "model/final_model/best_pickle_file.pkl")

            self.logger.info(best_model_pkl_file)
            self.logger.info("==========================Training  Completed==========================")
            return True
        except Exception as e:
            self.logger.info(e)
            self.logger.info("Error in selecting model.")
