#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:22:28 2024

@author: jczars
"""
# Imports
import csv
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Auxiliary functions
def create_folders(_save_dir, flag=1):
    """
    Create folders if they do not exist.
    
    Args:
        _save_dir (str): Path to the folder to be created.
        flag (int): Flag to indicate whether to overwrite the folder (1 for not allowing, will display an error if it exists).
    """
    if os.path.isdir(_save_dir):
        if flag:
            raise FileNotFoundError("Folders already exist: ", _save_dir)
        else:
            print('Folders already exist: ', _save_dir)
    else:
        os.mkdir(_save_dir)
        print('Created folder: ', _save_dir)

def read_print_csv(filename_csv):
    """
    Read and print rows from a CSV file.
    
    Args:
        filename_csv (str): Name of the CSV file to be read.
    """
    with open(filename_csv, 'r') as file:
        read_csv = csv.reader(file)
        for row in read_csv:
            print(row)

def add_row_csv(filename_csv, data):
    """
    Add a new row to a CSV file.
    
    Args:
        filename_csv (str): Name of the CSV file.
        data (list): Data to be inserted into the CSV file.
    """
    with open(filename_csv, 'a') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerows(data)

def load_data_train(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training data.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(rescale=1. / 255, validation_split=SPLIT_VALID)

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation'
    )

    return train_generator, val_generator

def load_data_train_aug(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training data with augmentation.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest',
        horizontal_flip=True,
        rescale=1./255,
        validation_split=SPLIT_VALID
    )

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation'
    )

    return train_generator, val_generator

def load_data_ttv(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training and validation data.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    val_dir = PATH_BD + '/Val/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(rescale=1. / 255)

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_generator = idg.flow_from_directory(
        directory=val_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    return train_generator, val_generator

def load_data_test(PATH_BD, K, BATCH, INPUT_SIZE):
    """
    Load test data.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        
    Returns:
        Generator: Test dataset.
    """
    test_dir = PATH_BD + '/Test/k' + str(K)
    print('Test directory: ', test_dir)

    idg = ImageDataGenerator(rescale=1. / 255)
    test_generator = idg.flow_from_directory(
        directory=test_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )
    return test_generator

def verifyGPU():
    """
    Verify and print the number of available GPUs.
    """
    print("##" * 30)
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("##" * 30)
    print('\n')

def figure_size(CATEGORIES):
    """
    Determine the figure size based on the number of categories.
    
    Args:
        CATEGORIES (list): List of categories.
        
    Returns:
        tuple: Figure size.
    """
    global fig_size
    tam = len(CATEGORIES)
    if tam > 20:
        w = 100
        h = 100
        fig_size = (w, h)
        print('Figure size: ', fig_size)
    else:
        fig_size = (10, 8)
        print('Figure size: ', fig_size)
    return fig_size

"""# Main"""

if __name__ == "__main__":
    help(read_print_csv)
    help(add_row_csv)
    help(load_data_train)
    help(load_data_test)
