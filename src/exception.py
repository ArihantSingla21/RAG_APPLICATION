import sys
from typing import Optional
from src.logger import logging



def error_message_detail(error,error_detail:sys):
    _,_,exc_tb= error_detail.exc_info() 
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))


class CustomException(Exception):
    """Custom exception class for the application."""
    
    def __init__(self, error_message: str, error_detail: Optional[sys.exc_info] = None):
        """
        Initialize the CustomException.
        
        Args:
            error_message: Human-readable error message
            error_detail: System error details (optional)
        """
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail if error_detail else sys.exc_info()

    def __str__(self):
        """Return string representation of the error."""
        return self.error_message
    
