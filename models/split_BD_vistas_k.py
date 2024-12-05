# -*- coding: utf-8 -*-
"""
Script for data processing and augmentation, structured into functions for dataset
creation, folder structure setup, image copying, and K-Fold splitting.

"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import glob
from sklearn.model_selection import StratifiedKFold
import yaml

def create_dataSet(path_data, csv_data, categories):
    """
    Creates a dataset in CSV format, listing file paths and labels.

    Parameters:
    - path_data (str): Path to the data.
    - csv_data (str): Path for saving the .csv file.
    - categories (list): List of categories or classes for the data.

    Returns:
    - DataFrame: DataFrame with file paths and labels.
    """
    data = pd.DataFrame(columns=['file', 'labels'])
    index = 0
    for category in tqdm(categories):
        path = os.path.join(path_data, category)
        filenames = os.listdir(path)
        for filename in filenames:
            data.loc[index] = [os.path.join(path, filename), category]
            index += 1
    data.to_csv(csv_data, index=False, header=True)
    data_csv = pd.read_csv(csv_data)
    print('\nPath to CSV data:', csv_data)
    print(data_csv.groupby('labels').count())
    return data

def create_folders(save_dir, flag=1):
    """
    Creates folders if they do not exist.

    Parameters:
    - save_dir (str): Path for the folder.
    - flag (int): Flag to rewrite folder. 1 for overwrite, 0 to skip.
    """
    if os.path.isdir(save_dir):
        if flag:
            raise FileNotFoundError("Folders already exist:", save_dir)
        else:
            print('Folders already exist:', save_dir)
    else:
        os.mkdir(save_dir)
        print('Created folders:', save_dir)

def kfold_split(data_csv, path, k, base_name):
    """
    Splits data into K folds using Stratified K-Fold and saves them as separate CSV files.

    Parameters:
    - data_csv (DataFrame): Data containing file paths and labels.
    - path (str): Path to save split data.
    - k (int): Number of folds.
    - base_name (str): Base name for saved CSV files.
    """
    labels = data_csv[['labels']]
    n = len(labels)
    print('Total data:', n)
    kfold = StratifiedKFold(n_splits=int(k), random_state=7, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(n), labels), 1):
        train_data = data_csv.iloc[train_idx]
        test_data = data_csv.iloc[test_idx]
        train_csv = os.path.join(path, f"{base_name}_trainSet_k{i}.csv")
        test_csv = os.path.join(path, f"{base_name}_testSet_k{i}.csv")
        train_data.to_csv(train_csv, index=False, header=True)
        test_data.to_csv(test_csv, index=False, header=True)

def copy_imgs(training_data, dst):
    """
    Copies images from source to destination.

    Parameters:
    - training_data (DataFrame): DataFrame containing image file paths.
    - dst (str): Destination path.
    """
    for file_path in training_data['file']:
        folder = os.path.basename(os.path.dirname(file_path))
        filename = os.path.basename(file_path)
        dst_folder = os.path.join(dst, folder)
        create_folders(dst_folder, flag=0)
        shutil.copy(file_path, os.path.join(dst_folder, filename))

def copy_img_k(path_csv, path_train, base_name, k, data_type='train'):
    """
    Copies images for each fold into separate folders.

    Parameters:
    - path_csv (str): Path to CSV files.
    - path_train (str): Path to training folder.
    - base_name (str): Base name for folders.
    - k (int): Number of folds.
    - data_type (str): Type of data ('train' or 'test').
    """
    for i in range(k):
        folder = os.path.join(path_train, f'k{i+1}')
        create_folders(folder, flag=0)
        csv_path = os.path.join(path_csv, f"{base_name}_{data_type}Set_k{i+1}.csv")
        data_imgs = pd.read_csv(csv_path)
        copy_imgs(data_imgs, folder)

def quantize(dst, categories, k, src, view, data_type='train'):
    """
    Counts images in each category for each fold and saves the results to a CSV file.

    Parameters:
    - dst (str): Path to destination.
    - categories (list): List of categories.
    - k (int): Number of folds.
    - output_csv (str): Path to the output CSV file for saving results.
    """
    quantization_data = []

    for i in range(k):
        fold_name = f'k{i+1}'
        folder = os.path.join(dst, fold_name)
        print(f"Quantizing fold {fold_name} at path: {folder}")

        for category in categories:
            images_path = glob.glob(os.path.join(folder, category, '*.png'))
            count = len(images_path)
            print(f"{category} in fold {fold_name}: {count} images")
            quantization_data.append({'Fold': fold_name, 'Category': category, 'Count': count})

    # Convert the quantization data to a DataFrame and save to CSV
    csv_name = f"split_{view}_{data_type}_k{k}.csv"
    output_csv = os.path.join(src, csv_name)
    quantization_df = pd.DataFrame(quantization_data)
    quantization_df.to_csv(output_csv, index=False)
    print(f"Quantization results saved to {output_csv}")


def copy_csv(src, dst_csv):
    """
    Copies CSV files from source to destination.

    Parameters:
    - src (str): Source directory for CSV files.
    - dst_csv (str): Destination directory for CSV files.
    """
    csv_files = glob.glob(os.path.join(src, "*.csv"))
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        shutil.copy(csv_file, os.path.join(dst_csv, filename))

def setup_structure(base_path):
    """
    Creates folder structure for training, testing, and CSV files.

    Parameters:
    - base_path (str): Base path for the structure.
    """
    create_folders(base_path, flag=0)
    for folder_name in ['Train', 'Test', 'csv']:
        create_folders(os.path.join(base_path, folder_name), flag=0)

def copy_base(src, new_base, base_name, categories, k, bd_src, view):
    """
    Sets up folder structure, creates dataset, splits data, and copies images.

    Parameters:
    - src (str): Source data path.
    - new_base (str): New base path.
    - base_name (str): Base name for datasets.
    - categories (list): List of categories.
    - k (int): Number of folds.
    """
    print('Creating folder structure...')
    setup_structure(new_base)
    print('Creating dataset CSV...')
    path_csv = os.path.join(new_base, 'csv')
    csv_data = os.path.join(path_csv, f"{base_name}.csv")
    data_csv = create_dataSet(src, csv_data, categories)
    print('Creating K-fold splits...')
    kfold_split(data_csv, path_csv, k, base_name)
    print('Copying training images...')
    copy_img_k(path_csv, os.path.join(new_base, 'Train'), base_name, k, 'train')
    quantize(os.path.join(new_base, 'Train'), categories, k, bd_src, view, "train")
    print('Copying testing images...')
    copy_img_k(path_csv, os.path.join(new_base, 'Test'), base_name, k, 'test')
    quantize(os.path.join(new_base, 'Test'), categories, k, bd_src, view, "test")

def run_split(params):
    """
    Main function to process data according to YAML configuration parameters.

    Parameters:
    - params (dict): Configuration parameters loaded from YAML.
    """
    #views = ['EQUATORIAL', 'POLAR']
    views = params['views']
    for view in views:
        src_view = os.path.join(params['base_dir'], view)
        categories = sorted(os.listdir(src_view))
        dst_view = os.path.join(params['base_dir'], f"{view}_{params['goal']}")
        create_folders(dst_view, flag=1)
        copy_base(src_view, dst_view, params['nm_new_base'], categories, params['k_folds'], 
                  params['base_dir'], view)

def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Dictionary with configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Argument parser for command-line configuration
    parser = argparse.ArgumentParser(description="Run the script with YAML configuration.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./preprocess/config_split.yaml",
        help="Path to the YAML configuration file. Defaults to 'config.yaml'."
    )
    args = parser.parse_args()
    params = load_config(args.config)
    run_split(params)
