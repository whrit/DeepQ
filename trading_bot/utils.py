import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + f'{abs(price):.2f}'

# Formats Currency
format_currency = lambda price: f'${abs(price):.2f}'

def show_train_result(result, val_position, initial_offset):
    """ Displays training results """
    if val_position == initial_offset or val_position == 0.0:
        logging.info(f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: USELESS  Train Loss: {result[3]:.4f}')
    else:
        logging.info(f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])}  Val Position: {format_position(val_position)}  Train Loss: {result[3]:.4f}')

def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results """
    if profit == initial_offset or profit == 0.0:
        logging.info(f'{model_name}: USELESS\n')
    else:
        logging.info(f'{model_name}: {format_position(profit)}\n')

def get_stock_data(stock_file):
    """Reads stock data from csv file and returns as numpy array"""
    df = pd.read_csv(stock_file)
    return df['Adj Close'].to_numpy()

def switch_tf_backend_device():
    """ Switches `keras` backend to use MPS (Metal Performance Shaders) if available,
    or CUDA if MPS is not available.

    Optimized computation on Apple silicon using MPS or NVIDIA GPUs using CUDA.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU is available: {len(gpus)} GPU(s) detected")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU available, using CPU")