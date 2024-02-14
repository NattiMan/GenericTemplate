import sys
import os
from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_object, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

num_cols = ['Population']
cat_cols = []
targ_col = 'MedHouseVal'
transformation_instance = DataTransformation('data/californiaHousePrice.csv.csv',numerical_columns=num_cols, categorical_columns=cat_cols, is_testing=False, training_data_path='Artifacts/trainData.csv')
train_array , test_array, _ = transformation_instance.initiate_data_transformation(target_column=targ_col)

model_training_instance = ModelTrainer()
print(model_training_instance.initiate_model_trainer(train_array, test_array))


print(train_array.shape)
print(test_array.shape)
