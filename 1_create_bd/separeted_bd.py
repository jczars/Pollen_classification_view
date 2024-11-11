
import shutil
import os, sys
import numpy as np
import cv2
import pandas as pd
from keras import models
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from keras import models

# Configuração do caminho para importar módulos personalizados
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print("Caminhos de sistema:", sys.path)

# Importação de módulos e funções
from models import get_data, utils, models_pre, models_train, reports_build

"""
1 - função para ler as imagens do dataset criar um df que contém o nome da classe e as quantidades de imagens e salvar em um csv;
2 - função para ler as classes vistas, as classes vistas são seis classes, três para a vista EQUATORIAL e tres para a vista POLAR;
3 - funcao que ler o csv e carregar as imagens, utilizando imageDategenerator;
4 - funcao que recebe as imagens carregadas  e fazer predições com um modelo treinado verificar qual classe o true label da 
    imagem pertence a qual vista[EQUATORIAL, POLAR] e salvar as previsões em um csv, o csv deve guardar o caminho da imagem, 
    caminho do ŕotulo previsto, o rotulo previsto, a probabilidade da previsão e a vista.
    Exibir os resultados na tela.
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
    
    y_preds = model.predict(test_data_generator, steps=nb_samples // batch_size + 1)
    
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
    
    vistas = []   # Lista para armazenar as vistas correspondentes
    classes = []  # Lista para armazenar as classes

    # Iterar sobre as linhas do DataFrame
    for i, row in df.iterrows():
        # Acessar a previsão de categoria e o caminho do arquivo
        vista = categories[row['y_pred']]  # Corrigido para acessar 'y_pred' da linha atual
        classe = row['file']

        # Extrair a vista ("EQUATORIAL" ou "POLAR") e a classe a partir do caminho do arquivo
        vt = vista.split('_')[0]         # Extrair "EQUATORIAL" ou "POLAR"
        classe = classe.split('/')[-2]    # Extrair a classe

        # Adicionar os valores às listas
        vistas.append(vt)
        classes.append(classe)

    # Atribuir as listas como novas colunas ao DataFrame
    df['vista'] = vistas
    df['classe'] = classes

    # Agrupar o DataFrame por 'vista' e 'classe' e contar o número de imagens em cada combinação
    quantidade_por_vista_classe = df.groupby(['vista', 'classe']).size().reset_index(name='quantidade')

    return df, quantidade_por_vista_classe

import pandas as pd

def quantificar_e_filtrar_classes0(df, limiar):
    """
    Quantifica a quantidade de classes por vista e filtra as linhas do DataFrame 
    que pertencem às classes que possuem quantidade maior ou igual ao limiar.
    
    :param df: DataFrame contendo as colunas ['file', 'classe', 'vista']
    :param limiar: valor mínimo de quantidade para filtrar as classes
    :return: DataFrame com as linhas filtradas que atendem ao limiar
    """
    # Quantificar classes por vistas
    quantidades_por_vista = df.groupby(['vista', 'classe']).size().reset_index(name='quantidade')

    # Filtrar as classes que possuem quantidade maior ou igual ao limiar
    classes_aceitas = quantidades_por_vista[quantidades_por_vista['quantidade'] >= limiar]
    
    # Criar uma lista das classes que atendem ao limiar
    classes_aceitas_list = classes_aceitas['classe'].unique()
    
    # Filtrar o DataFrame original com base nas classes aceitas
    df_filtrado = df[df['classe'].isin(classes_aceitas_list)]
    
    # Exibir o resultado
    print(f"\nClasses que atendem ao limiar ({limiar}):")
    print(classes_aceitas)
    
    return df_filtrado

def filter_by_class_limiar(df, limiar):
    """
    Filters classes in the dataframe based on the provided limiar and adds a summary of class distribution by vista.
    
    Parameters:
    - df (DataFrame): The dataframe containing 'file', 'classe', and 'vista'.
    - limiar (int): The minimum number of occurrences for a class to be retained.
    
    Returns:
    - DataFrame: A filtered dataframe with rows where class counts exceed the limiar, 
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
    categories, categories_vistas, model = initial(params)
    csv_data=create_dataSet(params['bd_src'], params['bd_dst'], categories) 
    data = pd.read_csv(csv_data)
    test_data_generator = get_data.load_data_test(data, params['input_shape']) 

    df_vistas, df_quantidade = predict_data_generator(test_data_generator, model, categories_vistas, 
                           params['batch_size'], verbose=2)
    df_vistas.to_csv(f"{params['bd_dst']}df_vistas.csv", index=False)
    df_quantidade.to_csv(f"{params['bd_dst']}df_qde_vistas.csv", index=False)
    print(df_vistas.head())
    print(df_quantidade)

    df_lm_vistas, df_summary_filtered = filter_by_class_limiar(df_vistas, params['limiar'])
    print(df_lm_vistas.head())
    print(df_summary_filtered)

    df_lm_vistas.to_csv(f"{params['bd_dst']}df_lm_vistas.csv", index=False)
    df_summary_filtered.to_csv(f"{params['bd_dst']}df_summary_filtered.csv", index=False)

    copy_images_by_vista(df_lm_vistas, params['bd_dst'])

params={'input_shape':(224,224),
        'batch_size':16,
        'limiar':20,
        'bd_src': "./BD/CPD1_Is_Rc/",
        'bd_dst':"./BD/CPD1_Dn_VTcr_220824/",
        'path_model': "/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/Reports/3_DenseNet201_sem_BI_5/models/3_DenseNet201_bestLoss_8.keras",
        'path_labels': "/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/BD/BI_5/labels/",
        'motivo': "Refazer a BD Vistas utilizando a base isolated",
        'date':"21/08/24"
        }

if __name__ == "__main__":
    # Exemplo de uso
    run(params)

