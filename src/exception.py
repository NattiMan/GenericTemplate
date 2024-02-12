#  This module will handle our custom exceptions

    

import sys

def get_error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    script_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f'An error has happened in the script {script_name} at line {line_number}, and the error is {str(error)}'
    return error_message


class CustomException(Exception):

    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = get_error_message_detail(error_message, error_detail=error_detail)


    def __str__(self):
        return self.error_message

        

