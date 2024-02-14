import sys
import os
from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_object, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

from src.pipelines.train_pipeline import TrainPipeline

t_path = 'data/californiaHousePrice.csv.csv'
targ_col = 'MedHouseVal'
num_cols = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
cat_cols = []

train_app = TrainPipeline(training_data_path=t_path, target_column=targ_col, numerical_columns=num_cols, categorical_columns=cat_cols, is_testing=False)
train_app.train()

