"""
This module contains a function that can be used to define 
the default properties of the logging mechanisms used by estimationpy.
There are two default loggers that log to both console and a file called 
``estimationpy.log`` that will be generated in the local directory where the
python interpreter is executed.

The module provides a function that takes as argument the logging levels
for both the console and the log file.

If you want you can create your own configuration for the loggers
without using this function. The following code snippet shows how 
to create a logger that uses only the console and that shows messages
down to the debug level::

  import logging
  from logging.config import dictConfig

  logging_config = dict(
        version = 1,
        formatters = {
            'f': {'format':
                  '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s | %(lineno)d | %(message)s'}
        },
        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'level': logging.DEBUG
            }
        },
        loggers = {
            'estimationpy': {
                'handlers': ['console'],
                'level': logging.DEBUG
            }
        }
    )
    
    # Configure the logger
    dictConfig(logging_config)


"""
import logging
import os
from logging.config import dictConfig

def configure_logger(log_level = logging.DEBUG, log_level_console = logging.ERROR, \
                     log_level_file = logging.WARNING):
    """
    The functions allows to configure some of the properties of the logging mechanism
    used by estimationpy. In particular the function allows to specify the log levels
    used by the different handlers implemented: one for the console and one on log files.
    
    The levels can be defined using integers from 0 to 50, but one should used the
    predefined levels provided by the standard logging module.
    By default the console logs in ERROR mode, the file logs in WARNING mode and the overall
    logger in ERROR mode.
    See  `Python logging <http://docs.python-guide.org/en/latest/writing/logging/>`_ 
    and `Logging HOWTO <https://docs.python.org/2/howto/logging.html>`_ for more 
    information about the logging module.
    
    The priority for the messages is
    
    
    +----------+---------+
    | name     |  level  |
    +==========+=========+
    | CRITICAL |  50     |
    +----------+---------+
    | ERROR    |  40     |
    +----------+---------+
    | WARNING  |  30     |
    +----------+---------+
    | INFO     |  20     |
    +----------+---------+
    | DEBUG    |  10     |
    +----------+---------+
    
    
    Please note that when you define a package logging level ``log_level`` equal to ERROR you 
    implicitly prevent any log message with level lower than ERROR to be displayed or saved
    in the log file.
    
    :param int log_level: Logging level for the whole package
    :param int log_level_console: Logging level specific for the messages on the console
    :param int log_level_file: Logging level specific for the messages in the log file
    """

    # Dictionary with details about the loggers
    logging_config = dict(
        version = 1,
        formatters = {
            'f': {'format':
                  '%(asctime)s | %(name)-30s | %(levelname)-8s | %(funcName)s | %(lineno)d | %(message)s'}
        },
        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'level': log_level_console
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'f',
                'level': log_level_file,
                'backupCount':0,
                'maxBytes': 1024*1024*5,
                'filename': os.path.join(os.path.abspath(os.curdir), "estimationpy.log")
            }
        },
        loggers = {
            'estimationpy': {
                'handlers': ['console', 'file'],
                'level': log_level
            }
        }
    )
    
    # Configure the logger
    dictConfig(logging_config)

    return
