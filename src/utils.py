import os 
import sys
import pickle 
from src.exception import CustomException
from src.logger import logging as log
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, object):
    # check if the file path exists before trying to dump a pickle
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    try:
        with open (file_path, 'wb') as obj_file_path:
            pickle.dump(object, obj_file_path)
        
        log.info("SUCCESSFULLY saved the object.")
        
    except Exception as e:
        log.info("The object could not be saved successfully.")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    report = {}
    try:
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]


            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            
        return report  

    except Exception as e:
        raise CustomException(e, sys)



class PipeInputReceiver:

    def __init__(self, target_column,testing_data_path, numerical_columns = None, categorical_columns = None, is_testing = False):
        self.testing_data_path = testing_data_path
        self.target_column = target_column
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns 
        self.is_testing = is_testing