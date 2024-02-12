# This module is used to log main events through the execution of the app

import logging
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True)

log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") 
LOG_FILE_PATH = os.path.join('logs', f'{log_file_name}.log')


logging.basicConfig(
    filename = LOG_FILE_PATH,
    encoding = 'utf-8',
    level = logging.INFO

)