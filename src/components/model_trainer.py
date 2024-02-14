
# # Our target here is to create a model trainer module which will create the model
# # The model trainer will accept training data and output the trained model


# import os
# import sys

# from dataclasses import dataclass
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from catboost import CatBoostRegressor
# from sklearn.metrics import r2_score

# from src.exception import CustomException
# from src.utils import evaluate_models, save_object
# from src.logger import logging as log

# @dataclass
# class ModelTrainerConfig:
#     trainedModelPath: str = os.path.join('Artifacts','trainedModel.pkl')

# class ModelTrainer:

#     def __init__(self):
#         self.trained_model_path = ModelTrainerConfig()

#     def initiate_model_training(self, train_array, test_array):  

#         try:
#             log.info("Step 2")

#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             log.info("Step 3")

#             models = {
#                     "Decision Tree": DecisionTreeRegressor(),
#                     "Random Forest": RandomForestRegressor(),
#                     "Gradient Boosting": GradientBoostingRegressor(),
#                     "Linear Regression": LinearRegression(),
#                     "XGBRegressor": XGBRegressor(),
#                     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                     "AdaBoost Regressor": AdaBoostRegressor(),
#                     }
#             parameters = {
#                     "Decision Tree": {
#                         'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                         # 'splitter':['best','random'],
#                         # 'max_features':['sqrt','log2'],
#                     },
#                     "Random Forest":{
#                         # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
#                         # 'max_features':['sqrt','log2',None],
#                         'n_estimators': [8,16,32,64,128,256]
#                     },
#                     "Gradient Boosting":{
#                         # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#                         'learning_rate':[.1,.01,.05,.001],
#                         'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#                         # 'criterion':['squared_error', 'friedman_mse'],
#                         # 'max_features':['auto','sqrt','log2'],
#                         'n_estimators': [8,16,32,64,128,256]
#                     },
#                     "Linear Regression":{},
#                     "XGBRegressor":{
#                         'learning_rate':[.1,.01,.05,.001],
#                         'n_estimators': [8,16,32,64,128,256]
#                     },
#                     "CatBoosting Regressor":{
#                         'depth': [6,8,10],
#                         'learning_rate': [0.01, 0.05, 0.1],
#                         'iterations': [30, 50, 100]
#                     },
#                     "AdaBoost Regressor":{
#                         'learning_rate':[.1,.01,0.5,.001],
#                         # 'loss':['linear','square','exponential'],
#                         'n_estimators': [8,16,32,64,128,256]
#                     }
                    
#                 }
#             log.info("Step 4")
#             model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, parameters)

#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
#             best_model = models[best_model_name]

#             log.info("Step 5")
#             if best_model_score < 0.6:
#                 raise CustomException("All models underperformed, hence no best model found.")
#             save_object(self.trained_model_path.trainedModelPath, best_model)
#             predicted = best_model.predict(X_test)
#             prediction_error = r2_score(y_test, predicted)
#             return prediction_error
        
#         except Exception as e:
#             raise CustomException(e, sys)      
       


            
import os
import sys
import pandas as pd
import numpy as np
import sklearn
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging as log
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('Artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        log.info("Successfully created an instance of ModelTrainerConfig.")
    def initiate_model_trainer (self, train_array, test_array):
        log.info("Starting to initiate model trainer.")
        try:
            X_train, y_train = train_array[:1000, :-1], train_array[:1000, -1]  # Extracting features and target for training
            X_test, y_test = test_array[:1000, :-1], test_array[:1000, -1]     # Extracting features and target for testing


            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }


            params={
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            

            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            log.info("Successfully created the model report.")
            best_model_score = max(sorted(list(model_report.values())))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            log.info("Successfully created the model report.")
            log.info(f"Best model score is {best_model_score}")

            
            save_object(file_path = self.model_trainer_config.trained_model_path, object = best_model)
            log.info("Successfully saved the prediction model.")

            predicted_result = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted_result)

            return r2_square 


            
        except Exception as e:
            raise CustomException(e,sys)








