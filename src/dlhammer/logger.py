import os
import logging

logger = logging.getLogger('dlhammer')


def get_logfile(config):
    return os.path.join(config.OUTPUT_DIR, config.LOG_FILE)


def bootstrap_logger(logfile=None, fmt=None):
    global logger
    if fmt is None:
        fmt = '%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)

    if logfile is not None:
        formatter = logging.Formatter(fmt)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
