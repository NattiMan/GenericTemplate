import sys
import os
from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_object, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipelines.predict_pipeline import PredictPipeline
import pandas as pd



t_path = 'data/californiaHousePrice.csv.csv'
pred_df=pd.read_csv(t_path)




predict_pipe_instance = PredictPipeline()

output = predict_pipe_instance.predict(features=pred_df)

print(output)