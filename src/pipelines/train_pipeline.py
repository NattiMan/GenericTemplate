import sys
import os
from dataclasses import dataclass

from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_object, load_object

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Train pipeline should receive the inputs required from the user and return the trained model
# Let's start with data ingestion first

@dataclass
class TrainPipelineConfig:
    train_pipeline_preprocessor_path: str = os.path.join ('Artifacts', 'train_preprocessor')


class TrainPipeline:

    
    def __init__(self, training_data_path, target_column, numerical_columns = None, categorical_columns = None, is_testing = False):

        self.train_pipeline_configuration_instance = TrainPipelineConfig()
        self.training_data_path = training_data_path
        # self.testing_data_path = testing_data_path
        self.target_column = target_column
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns 
        self.is_testing = is_testing

    def train(self):
        data_ingestor = DataIngestion()        
        training_data_path, testing_data_path = data_ingestor.initiate_data_ingestion(self.training_data_path)
        log.info("Data ingestion has been successfully completed.")
        data_transformer = DataTransformation(
            testing_data_path,
            self.numerical_columns,
            self.categorical_columns,
            self.is_testing,
            training_data_path
        )
        # training_data_transforming_object = data_transformer.get_data_transform_object()
        train_array, test_array, _ = data_transformer.initiate_data_transformation(self.target_column)
        
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_array = train_array, test_array = test_array)

        return (self.train_pipeline_configuration_instance.train_pipeline_preprocessor_path)



















num_cols = ['Population']
cat_cols = []
targ_col = 'MedHouseVal'
transformation_instance = DataTransformation('data/californiaHousePrice.csv.csv',numerical_columns=num_cols, categorical_columns=cat_cols, is_testing=False, training_data_path='Artifacts/trainData.csv')
train_array , test_array, _ = transformation_instance.initiate_data_transformation(target_column=targ_col)

model_training_instance = ModelTrainer()
print(model_training_instance.initiate_model_trainer(train_array, test_array))


