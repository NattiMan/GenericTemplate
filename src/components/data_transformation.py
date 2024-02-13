
import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.logger import logging as log
from src.utils import save_object
from src.exception import CustomException
# This module takes the csv files paths and returns transformed arrays

class DataTransformationConfig:
    transformationProcessor_path: str = os.path.join('Artifacts', 'transformationProcessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_instance = DataTransformationConfig()
        log.info("DataTransformation instance created successfully.")


    def get_data_transformer(self, numerical_col = [], categorical_col = []):
        log.info("Now starting the method get_data_transformer.")
        
        self.numerical_col = numerical_col
        self.categorical_col = categorical_col

        try:
            num_pipeline = Pipeline(
            steps = [
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
                 ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_col),
                    ('cat', cat_pipeline, categorical_col)
                ]
            )

            save_object(self.data_transformation_instance.transformationProcessor_path, preprocessor)        

            return (
                preprocessor
            )
        except Exception as e:
            log.info("An error has happened while getting the data transformation ")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path, target_column):
        try:
            self.target_column = target_column

            self.preprocessor = self.get_data_transformer()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # When we transform we dont transform the target column

            features_train_dataset = train_df.drop(columns=self.target_column, axis=1)
            features_test_dataset = test_df.drop(columns=self.target_column, axis=1)

            target_test = test_df[target_column]
            target_train = train_df[target_column]

            scaled_train_dataset = self.preprocessor.fit_transform(features_train_dataset)
            scaled_test_dataset = self.preprocessor.transform(features_test_dataset)

            train_arr = np.c_[scaled_train_dataset, np.array(target_train)]
            test_arr = np.c_[scaled_test_dataset, np.array(target_test)]
            log.info("Successfully reconstructed the train_array and test_array.")
    
            return(
                train_arr,
                test_arr,
                self.data_transformation_instance.transformationProcessor_path
            )
        except Exception as e:
            log.info("Failed to run the method intitate data transformation.")
            raise CustomException(e, sys)



