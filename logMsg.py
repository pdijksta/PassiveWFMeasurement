import sys
from datetime import datetime as dt
import logging
from logging.handlers import RotatingFileHandler

# Copied from Eugenio Ferrari's wsServer

def get_logger(logfilename, title):
    try:
        handler = RotatingFileHandler(filename=logfilename,
                                      mode='a',
                                      maxBytes=5 * 1024 * 1024,
                                      backupCount=1,
                                      delay=0)
        handler2 = logging.StreamHandler(sys.stdout)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s:%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[handler, handler2])

        print('Logging file: %s' % logfilename)
    except Exception as e:
        print('Cannot write logging file...')
        print(e)
    logger = logging.getLogger(title)
    return logger

def logMsg(msg='', logger=None, style='I'):
    if logger:
        if style == 'I':
            logger.info(msg)
        elif style == 'W':
            logger.warning(msg)
        elif style == 'E':
            logger.error(msg)
        elif style == 'C':
            logger.critical(msg)
        else:
            logger.info(msg)
    else:
        time_str = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        if style == 'I':
            print('%s INFO: %s' % (time_str, msg))
        elif style == 'W':
            print('%s WARNING: %s' % (time_str, msg))
        elif style == 'E':
            print('%s ERROR: %s' % (time_str, msg))
        elif style == 'C':
            print('%s CRITICAL: %s' % (time_str, msg))
        else:
            print('%s %s' % (time_str, msg))

class LogMsgBase:
    def logMsg(self, msg, style='I'):
        return logMsg(msg, self.logger, style)

