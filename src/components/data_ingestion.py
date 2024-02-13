# This module takes the overall data source and breaks it down into training testing and raw data storages


import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging as log
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('Artifacts', 'trainData.csv')
    test_data_path:str = os.path.join('Artifacts', 'testData.csv')
    raw_data_path:str = os.path.join('Artifacts', 'rawData.csv')


class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_instance = DataIngestionConfig()
        
    def initiate_data_ingestion(self, file_path):

      try:
            log.info("Starting data ingestion.")
            log.info("DataIngestion object created successfully.")
            input_data = pd.read_csv(file_path)
            train_df, test_df = train_test_split(input_data, random_state=42, test_size=0.3)
            log.info("Successfully splitted train and test datasets.")
            train_df.to_csv(self.data_ingestion_instance.train_data_path)
            test_df.to_csv(self.data_ingestion_instance.test_data_path)
            log.info("train data and test data files are created.")
            return (
                self.data_ingestion_instance.train_data_path,
                self.data_ingestion_instance.test_data_path
            )
      except Exception as e:
          raise CustomException(e, sys)
          









