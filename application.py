import sys
import os
from src.logger import logging as log
from src.exception import CustomException
from src.utils import save_object, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


