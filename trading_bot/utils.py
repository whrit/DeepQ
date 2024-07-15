import os
import math
import logging

import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def switch_k_backend_device():
    """ Switches `keras` backend to use MPS (Metal Performance Shaders) if available,
    or CUDA if MPS is not available.

    Optimized computation on Apple silicon using MPS or NVIDIA GPUs using CUDA.
    """
    if K.backend() == "tensorflow":
        logging.debug("Checking for MPS support")
        if tf.config.experimental.list_physical_devices('MPS'):
            logging.debug("MPS support found, switching to MPS")
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            os.environ["TF_MPS_ENABLED"] = "1"
            physical_devices = tf.config.experimental.list_physical_devices('MPS')
            try:
                # Set memory growth for MPS devices
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except Exception as e:
                # Invalid device or cannot modify virtual devices once initialized
                logging.error(f"Error while setting MPS configuration: {e}")
        else:
            logging.debug("MPS support not found, checking for CUDA devices")
            if tf.config.experimental.list_physical_devices('GPU'):
                logging.debug("CUDA device found, switching to CUDA")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first CUDA device
                physical_devices = tf.config.experimental.list_physical_devices('GPU')
                try:
                    # Set memory growth for CUDA devices
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                    # Optionally, you can limit GPU memory usage
                    # tf.config.experimental.set_virtual_device_configuration(
                    #     physical_devices[0],
                    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                    # )
                except Exception as e:
                    logging.error(f"Error while setting CUDA configuration: {e}")
            else:
                logging.warning("Neither MPS nor CUDA support found, using default CPU backend")