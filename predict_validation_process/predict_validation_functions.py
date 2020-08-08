from logger.logconfig import getlogger
from dbConnection.mongo import DatabaseConnect
import pandas as pd
import json
import pickle

# initilizing DB connection
db_conn = DatabaseConnect()

# class for creating all prediction methods
class PredictValidationFunctions:
    def __init__(self, df):

        # initializing logger file for prediction
        self.logger = getlogger(
            __name__, './logger/prediction_logs.log', consoleHandlerrequired=True)

        # schema json file which is getting used to validate predict csv data
        self.schema_path = 'schema_prediction.json'
        self.df = df
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
                            "Type of columns are not similar in the dataset and schema. Please check the datatype of uploaded csv.")
                except Exception as e:
                    self.logger.info(e)
                    self.logger.info(
                        "column type issue in schema_prediction.json")
                    raise e
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

        # dropping row which has any value zero
        self.df.dropna(axis=0, how="any", inplace=True)
        self.logger.info("Null Values Count After :")
        self.logger.info(str(self.df.isna().sum()))

    # function to load best_pickle file, predict data and store csv and predicted data both into database
    def predictValues(self):
        self.logger.info("----------predictValues Starts----------")
        X = self.df.drop(columns=['id'], axis=1)
        self.logger.info(X.shape)
        pid = self.df['id']
        dfrows = X.shape[0]
        value_lst = []
        result_list = []
        try:
            filename = 'models/final_model/best_pickle_file.pkl'
            model = pickle.load(open(filename, 'rb'))
            for row in range(dfrows):
                value_lst.clear()
                for value in X.iloc[row]:
                    value_lst.append(value)
                predict_value = model.predict([value_lst])

                # storing all the predictions(pid, cancer_type) in list
                result_list.append(
                    {"pid": int(pid[row]), "cancer_type": int(predict_value[0])})

            df = pd.DataFrame(result_list, index=None)

            # Storing all the predicted values into Database
            db_conn.storePredictedResult(df)

            # Store predicted data into DB
            # db_conn.storePredictCSVToDB(self.df)
            self.logger.info(
                "==========================Prediction Completed==========================")
            return True
        except Exception as e:
            self.logger.info(e)
            raise e
