
import shutil
import os, sys
import numpy as np
import pandas as pd
from keras import models
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from keras import models
import  argparse, yaml

# Adiciona o diretório atual (pwd) ao PYTHONPATH
sys.path.insert(0, os.getcwd())
# Import custom modules and functions
from models import get_data, utils, del_folders_limiar, listar_vistas, sound_test_finalizado

"""
Script functions:

1 - Function to read images from the dataset, create a DataFrame containing class names and image counts, and save it to a CSV.
2 - Function to read the viewed classes, which include six classes: three for the EQUATORIAL view and three for the POLAR view.
3 - Function to load the CSV and images, using ImageDataGenerator.
4 - Function to make predictions using a trained model, identify the true label’s view (EQUATORIAL or POLAR), and save predictions in a CSV.
    The CSV should store the image path, predicted label path, predicted label, prediction probability, and view.
    Display results on the screen.
"""

def initial(params):
    """
    Initialize environment, load categories and labels, and set up model for predictions.

    Parameters:
    - params (dict): Dictionary containing configuration with the following keys:
        - 'bd_src' (str): Source directory path containing image categories.
        - 'path_labels' (str): Path to the labels directory.
        - 'bd_dst' (str): Destination directory path for output files.
        - 'path_model' (str): Path to the trained model file.
        - 'motivo' (str): Reason or purpose of the run.
        - 'date' (str): Date of the run or current date as a string.

    Returns:
    - tuple: (categories, categories_vistas, model)
        - categories (list): List of sorted categories in the source directory.
        - categories_vistas (list): List of sorted label categories.
        - model: Loaded machine learning model.
    """
    # Load categories from the source directory
    try:
        categories = sorted(os.listdir(params['bd_src']))
    except FileNotFoundError as e:
        print(f"Error: Source directory '{params['bd_src']}' not found. {e}")
        return None, None, None

    # Load categories from the labels directory
    try:
        categories_vistas = sorted(os.listdir(params['path_labels']))
        print("categories labels loaded:", categories_vistas)
    except FileNotFoundError as e:
        print(f"Error: Labels directory '{params['path_labels']}' not found. {e}")
        return None, None, None

    # Ensure destination folder exists
    utils.create_folders(params['bd_dst'], flag=1)

    # Prepare and save CSV header with metadata
    _csv_head = os.path.join(params['bd_dst'], 'head_.csv')
    metadata = [
        ["modelo", "path_labels", "motivo", "data"],
        [params['path_model'], params['path_labels'], params['motivo'], params.get('date', str(datetime.now().date()))]
    ]
    utils.add_row_csv(_csv_head, metadata)
    print("Metadata saved to CSV:", metadata)

    # Load the model for predictions
    try:
        model = models.load_model(params['path_model'])
        model.summary()  # Print model summary
    except Exception as e:
        print(f"Error loading model from '{params['path_model']}': {e}")
        return None, None, None
    

    return categories, categories_vistas, model

def create_dataSet(_path_data, _csv_data, _categories=None):
    """
    Creates a dataset in CSV format, listing image paths and labels.

    Parameters:
    - _path_data (str): Path to the root data directory containing class subdirectories.
    - _csv_data (str): Path with file name to save the CSV (e.g., 'output_data.csv').
    - _categories (list, optional): List of class names to include. If None, all subdirectories are used.

    Returns:
    - pd.DataFrame: DataFrame containing file paths and labels.
    """
    _csv_data=f"{_csv_data}data.csv"
    data = pd.DataFrame(columns=['file', 'labels'])
    cat_names = os.listdir(_path_data) if _categories is None else _categories
    c = 0

    for j in tqdm(cat_names, desc="Processing categories"):
        pathfile = os.path.join(_path_data, j)
        
        # Check if the path is a directory
        if not os.path.isdir(pathfile):
            print(f"Warning: {pathfile} is not a directory, skipping.")
            continue

        filenames = os.listdir(pathfile)
        for i in filenames:
            # Full file path
            file_path = os.path.join(pathfile, i)
            
            # Check if it's a valid file (e.g., image file)
            if os.path.isfile(file_path):
                data.loc[c] = [file_path, j]
                c += 1

    # Save DataFrame to CSV
    try:
        data.to_csv(_csv_data, index=False, header=True)
        print(f'\nCSV saved successfully at: {_csv_data}')
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None

    # Read and print summary from the CSV
    try:
        data_csv = pd.read_csv(_csv_data)        
        # Create and save the summary CSV with counts of images per label
        _summary_csv = _csv_data.replace('.csv', '_summary.csv')
        label_counts = data.groupby('labels').size().reset_index(name='count')
        label_counts.to_csv(_summary_csv, index=False, header=True)
        print(data_csv.groupby('labels').count())

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    return _csv_data 

def predict_data_generator(test_data_generator, model, categories, batch_size, verbose=2):
    """
    Generates predictions and evaluation metrics (accuracy, precision, recall, fscore, kappa) for test data.
    Returns two DataFrames: one for correct predictions and one for incorrect predictions.

    Parameters:
        test_data_generator (ImageDataGenerator): Image Data Generator containing test data.
        model (keras.Model): Trained model used for prediction.
        categories (list): List of image class names.
        batch_size (int): Number of samples per batch.
        verbose (int): Verbosity level (0, 1, or 2). Default is 2.

    Returns:
        tuple: y_true (true labels), y_pred (predicted labels), 
               df_correct (DataFrame containing correctly classified samples), 
               df_incorrect (DataFrame containing incorrectly classified samples).
    """
    filenames = test_data_generator.filenames
    df = pd.DataFrame(filenames, columns=['file'])
    confidences = []
    nb_samples = len(filenames)
    print('Predicting unlabeled data...', nb_samples)
    print(f'Batch size: {batch_size}')
    #, steps=nb_samples // batch_size + 1
    y_preds = model.predict(test_data_generator)
    
    for prediction in y_preds:
        confidence = np.max(prediction)
        confidences.append(confidence)
        if verbose == 1:
            print(f'Prediction: {prediction}, Confidence: {confidence}')
    
    y_pred = np.argmax(y_preds, axis=1)
    
    if verbose == 2:
        print(f'Size y_pred: {len(y_pred)}')
    
    df['y_pred'] = y_pred
    df['confidence'] = confidences
    df['predicted_label'] = [categories[i] for i in y_pred]
    
    vistas = []   # List to store views
    classes = []  # List to store classes

    # Iterar sobre as linhas do DataFrame
    for i, row in df.iterrows():
        # Access category prediction and file path
        vista = categories[row['y_pred']]  # Corrigido para acessar 'y_pred' da linha atual
        classe = row['file']

        # Extract the view ("EQUATORIAL" or "POLAR") and class from the file path
        vt = vista.split('_')[0]        
        classe = classe.split('/')[-2]    

        vistas.append(vt)
        classes.append(classe)

    df['vista'] = vistas
    df['classe'] = classes

    # Group DataFrame by 'vista' and 'classe' and count the number of images per combination
    quantidade_por_vista_classe = df.groupby(['vista', 'classe']).size().reset_index(name='quantidade')

    return df, quantidade_por_vista_classe

def filter_by_class_limiar(df, limiar):
    """
    Filters classes in the dataframe based on the provided threshold and adds a summary of class distribution by view.
    
    Parameters:
    - df (DataFrame): The dataframe containing 'file', 'classe', and 'vista'.
    - limiar (int): The minimum number of occurrences for a class to be retained.
    
    Returns:
    - DataFrame: A filtered dataframe with rows where class counts exceed the threshold, 
      and a summary DataFrame appended at the end.
    """
    # Quantify the number of occurrences of each class per vista
    df_summary = df.groupby(['vista', 'classe']).size().reset_index(name='quantidade')

    # Filter classes where the count exceeds the limiar
    df_summary_filtered = df_summary[df_summary['quantidade'] >= limiar]

    # Get the list of classes to retain based on the filtered summary
    classes_to_retain = df_summary_filtered['classe'].tolist()

    # Filter the original dataframe to keep only rows where 'classe' is in the retained list
    df_filtered = df[df['classe'].isin(classes_to_retain)]

    # Return the filtered dataframe and the summary dataframe at the end
    return df_filtered, df_summary_filtered

def copy_images_by_vista(df_lm_vistas, destination_dir):
    """
    Copies images from the source directory to subfolders named after their class ('EQUATORIAL' or 'POLAR') 
    in the destination directory. Each class will be placed in a subfolder under either 'EQUATORIAL' or 'POLAR'.
    
    Parameters:
    - df_lm_vistas (DataFrame): DataFrame containing columns ['file', 'vista'], with 'file' as image paths and 
      'vista' as either 'EQUATORIAL' or 'POLAR'.
    - destination_dir (str): The directory where the images should be copied, in subfolders 'EQUATORIAL' and 'POLAR'.
    
    Returns:
    - None
    """
    # Ensure the destination directories exist
    equatorial_dir = os.path.join(destination_dir, 'EQUATORIAL')
    polar_dir = os.path.join(destination_dir, 'POLAR')
    
    os.makedirs(equatorial_dir, exist_ok=True)
    os.makedirs(polar_dir, exist_ok=True)

    # Iterate through the DataFrame and copy images based on the 'vista' column
    for _, row in df_lm_vistas.iterrows():
        # Get file path and vista from DataFrame
        file_path = row['file']  # full path of the image
        vista = row['vista']

        # Extract class from the file path (second-to-last directory in the path)
        class_name = file_path.split('/')[-2]  # Class is the second-to-last element in the path

        # Determine the destination folder based on vista
        if vista == 'equatorial':
            destination_folder = os.path.join(equatorial_dir, class_name)
        elif vista == 'polar':
            destination_folder = os.path.join(polar_dir, class_name)
        else:
            continue  # Skip if vista is not valid

        # Ensure the class subdirectory exists
        os.makedirs(destination_folder, exist_ok=True)

        # Determine the destination file path
        destination_path = os.path.join(destination_folder, os.path.basename(file_path))

        # Copy the image to the appropriate directory
        try:
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

def run(params):
    image_size=params['image_size'] 
    input_shape=(image_size, image_size)
    categories, categories_vistas, model = initial(params)
    csv_data=create_dataSet(params['bd_src'], params['bd_dst'], categories) 
    data = pd.read_csv(csv_data)
    test_data_generator = get_data.load_data_test(data, input_shape) 

    df_vistas, df_quantidade = predict_data_generator(test_data_generator, model, categories_vistas, 
                           params['batch_size'], verbose=2)
    df_vistas.to_csv(f"{params['bd_dst']}df_vistas.csv", index=False)
    df_quantidade.to_csv(f"{params['bd_dst']}df_qde_vistas.csv", index=False)
    print("Fase 1: Criando BD")
    print(df_vistas.head())
    print(df_quantidade)

    df_lm_vistas, df_summary_filtered = filter_by_class_limiar(df_vistas, params['limiar'])
    print("Fase 2: filtrar pro limiar")
    print(df_lm_vistas.head())
    print(df_summary_filtered)

    df_lm_vistas.to_csv(f"{params['bd_dst']}df_lm_vistas.csv", index=False)
    df_summary_filtered.to_csv(f"{params['bd_dst']}df_summary_filtered.csv", index=False)

    copy_images_by_vista(df_lm_vistas, params['bd_dst'])

    params_del={'tipo':'/*.png',
        'bd_dst': params['bd_dst'],
        'vistas':['EQUATORIAL','POLAR'],
        'flag': 1, #0-não deleta, 1-deleta
        'limiar': params['limiar'] # menor quantidade de exemplos por classes
        }
    
    del_folders_limiar.del_vistas(params_del)

    params_list={
        'vistas':params_del['vistas'],
        'save_dir': params['bd_dst'],
        'path_data': params['bd_dst'],
        'tipo': 'png',
        'version':3  
    }
    listar_vistas.run(params_list)

    EQUATORIAL_dir = os.path.join(params['bd_dst'], 'EQUATORIAL')
    POLAR_dir = os.path.join(params['bd_dst'], 'POLAR')

    figEQ = utils.graph_img_cat(EQUATORIAL_dir)
    figPL = utils.graph_img_cat(POLAR_dir)
    save_dir = params['bd_dst']
    if save_dir:
        figEQ.savefig(os.path.join(save_dir, 'img_cat_EQAUTORIAL.jpg'))
        figPL.savefig(os.path.join(save_dir, 'img_cat_POLAR.jpg'))

def parse_args():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Resize images and organize them by category.")
    
    parser.add_argument('--config', type=str, help="Path to the configuration YAML file.")
    
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = parse_args()

    # Load configuration from YAML file
    config_file = args.config if args.config else './preprocess/config_separeted.yaml'
    params = load_config(config_file)

    #run(params)
    #python 1_create_bd/separeted_bd.py --config 1_create_bd/config_separeted.yaml
    
    debug = True
    
    if debug:
        # Run the training and evaluation process in debug mode
        run(params)
        sound_test_finalizado.beep(2)
    else:        
        try:
            # Run the training and evaluation process and send success notification
            run(params)
            message = '[INFO] successfully!'
            print(message)
            sound_test_finalizado.beep(2, message)
        except Exception as e:
            # Send error notification if the process fails            
            message = f'[INFO] with ERROR!!! {str(e)}'
            print(message)
            sound_test_finalizado.beep(2, message)

