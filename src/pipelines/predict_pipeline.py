import sys
import os
import pandas as pd
import sklearn
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','transformationProcessor2.pkl')
            print("Now Loading the preprocessor and prediction model.")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            print("Successfully loaded the preprocessor and prediction model.")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


