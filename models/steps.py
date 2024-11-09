import sys

import pandas as pd
from models import models_pre, models_train, utils
from models import get_data

def step_0(conf, _labels, labels_categories):
    """
    Sets up the environment for the pollen classification experiment.

    Args:
        conf (dict): Configuration dictionary containing:
            - id_test (str): Identifier for the test.
            - model (str): Model name to be used.
            - aug (str): Data augmentation method.
            - base (str): Base folder name.
            - labels (list): List of labels for the images.
            - root (str): Root directory for results.
            - path_base (str): Path to the base dataset.

        _labels (list): List of labels corresponding to the images.
        labels_categories (list): List of categories for classification.

    Returns:
        dict: A dictionary containing paths to the datasets and configuration information.
    """
    # Verificação de categorias
    if not labels_categories:
        raise ValueError("The classification categories list cannot be empty.")
    
    test_id = conf['id_test']
    model_name = conf['model']
    augmentation = conf['aug']
    base_name = conf['base']
    root_path = conf['root']
    path_base = conf['path_base']

    print('\n[STEP 0].1 - Create the root folder')
    utils.create_folders(root_path, flag=0)

    print('\n[STEP 0].2 - Create the test folder')
    test_folder_name = f"{test_id}_{model_name}_{augmentation}_{base_name}"
    test_folder_path = f"{root_path}/{test_folder_name}"
    utils.create_folders(test_folder_path, flag=0)

    print('\n[STEP 0].3 - Create the Train folder')
    save_dir_train, train_id = utils.criarTestes(test_folder_path, test_folder_name)
    print(f'Save directory for training: {save_dir_train}, ID: {train_id}')

    print('\n[STEP 0].4 - Create the pseudo_csv folder')
    pseudo_csv_path = f"{save_dir_train}/pseudo_csv"
    utils.create_folders(pseudo_csv_path, flag=0)

    print('\n[STEP 0].5 - Create the CSV')
    csv_data_path = f"{path_base}/{base_name}.csv"
    print('CSV data path:', csv_data_path)
    data_csv = utils.create_dataSet(_labels, csv_data_path, labels_categories)

    num_labels = len(data_csv)
    print('Total number of labeled data:', num_labels)

    print('\n[STEP 0].6 - Split the data')
    path_train, path_val, path_test = get_data.splitData(data_csv, root_path, base_name)

    return {
        'path_train': path_train,
        'path_val': path_val,
        'path_test': path_test,
        'save_dir_train': save_dir_train,
        'pseudo_csv': pseudo_csv_path,
        'size_of_labels': num_labels
    }
def step_1(conf, labels_categories, return_0, _tempo):
    """
    Step 1: Load training data, train the model, and evaluate the model's performance.

    Args:
        conf (dict): Configuration dictionary containing:
            - img_size (int): Size of the input images.
            - aug (str): Data augmentation method.
            - type_model (str): Type of model to be trained (e.g., 'pre', 'att').
        
        labels_categories (list): List of categories (labels) for classification.
        
        return_0 (dict): Contains paths to datasets and additional configuration info returned from step 0.
        
        _tempo (float): Time tracking variable to monitor training duration.

    Returns:
        tuple: A tuple containing the trained model instance and a dictionary with evaluation metrics.
    """
    print('Step 1: Initiating the training process.')

    # Load training and validation data
    print('\n[Step 1.1] - Load training data')
    training_data = pd.read_csv(return_0['path_train'])
    validation_data = pd.read_csv(return_0['path_val'])
    img_size = conf['img_size']
    input_size = (img_size, img_size)
    
    train, val = get_data.load_data_train(training_data, validation_data, conf['aug'], input_size)

    # Instantiate the model based on the specified type
    print('\n[Step 1.2] - Instantiate the model')
    model_map = {
        'imagenet': models_pre.hyper_model,
    }

    model_type = conf['type_model']
    if model_type not in model_map:
        raise ValueError(f"Invalid model type '{model_type}'. Expected one of {list(model_map.keys())}.")
    
    model_inst = model_map[model_type](conf, len(labels_categories))

    # Train the model
    print(f'\n[INFO] - Training the model. Time elapsed: {_tempo}')
    model_inst, num_epoch, str_time, end_time, delay = models_train.fitModels(conf, return_0, _tempo, model_inst, train, val)

    # Evaluate the model
    print(f'\n[INFO] - Evaluating the model. Time elapsed: {_tempo}')
    test_data = pd.read_csv(return_0['path_test'])
    test = get_data.load_data_test(test_data, input_size)

    metrics = reports_gen.reports_build(conf, test, model_inst, labels_categories, _tempo, return_0)
    
    return model_inst, {
        'test_loss': metrics['test_loss'],
        'test_accuracy': metrics['test_accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'fscore': metrics['fscore'],
        'kappa': metrics['kappa'],
        'num_epoch': num_epoch,
        'str_time': str_time,
        'end_time': end_time,
        'delay': delay
    }
