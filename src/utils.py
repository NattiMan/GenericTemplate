import os 
import sys
import sklearn.mo
import pickle 
from src.exception import CustomException
from src.logger import logging as log


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

def load_object(object_path):
    try:
        with open (object_path, 'rb') as obj:
            pickle.load(obj)
        log.info("SUCCESSFULLY loaded the pickle object.")
        
    except Exception as e:
        log.info("Pickle object could not be loaded.")
        raise CustomException(e, sys)
    
def evaluate_model(test_data_path, model):
    ''' This function takes train and test data along with the models and their params
    and returns a dictionary of the model name and their r2_square errors
    '''
    




