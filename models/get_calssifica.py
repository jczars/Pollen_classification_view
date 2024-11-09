import sys
<<<<<<< HEAD
import gc, os
=======
import gc
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
import pandas as pd

# Add the path for the modules
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)

from models import get_data
from models import utils
<<<<<<< HEAD
from models import reports_build
=======
from models import reports
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5

def classificaImgs(conf, _path_model, _tempo, model_inst, _pseudo_csv,
                   CATEGORIES, tipo='png', verbose=1):
    """
    Classifies unlabeled images and generates pseudo-labels.

    Steps performed:
    1. Load model weights.
    2. Instantiate the model with the loaded weights.
    3. Load unlabeled images from the specified CSV.
    4. Generate pseudo-labels for the unlabeled images.
    5. Save the dataset of unlabeled images, including the timestamp.

    Parameters:
        conf: Configuration dictionary containing model parameters.
        _path_model: Path to the model weights.
        _tempo: Time variable for tracking iterations.
        model_inst: The model instance used for classification.
        _pseudo_csv: Path to the CSV containing pseudo-labels.
        CATEGORIES: List of image classes.
        tipo: Type of images ('png' by default).
        verbose: Verbosity level for debugging output.

    Returns:
        List of classified unlabeled data with pseudo-labels.
    """
    print('[classificaImgs].1 - Loading model weights')
    path = _path_model + conf['model'] + '_bestLoss_' + str(_tempo) + '.keras'
    print(path)

    print('[classificaImgs].2 - Instantiating the model with weights')
    model_inst.load_weights(path)

    print('[classificaImgs].3 - Loading unlabeled images from CSV')
    # Load unlabeled data
    if _tempo == 0:
        unalbels_generator = get_data.load_unlabels(conf)
    else:
        _cs_uns_ini = _pseudo_csv + '/unlabelSet_T' + str(_tempo) + '.csv'
        df_uns_ini = pd.read_csv(_cs_uns_ini)
        if len(df_uns_ini) > 0:
            print(df_uns_ini.head())
            print(f"\ntempo {_tempo}, read _cs_uns_ini{_cs_uns_ini}")
            unalbels_generator = get_data.load_data_test(df_uns_ini, input_size=(224, 224))
        else:
            return df_uns_ini

    print('[classificaImgs].4 - Generating pseudo-labels for unlabeled data')
<<<<<<< HEAD
    data_uns = reports_build.predict_unlabels_data_gen(conf, unalbels_generator, model_inst, CATEGORIES)
=======
    data_uns = reports.predict_unlabels_data_gen(conf, unalbels_generator, model_inst, CATEGORIES)
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
    print(f"data_uns, {len(data_uns)}")

    return data_uns


<<<<<<< HEAD
def selec(conf, data_uns_ini, _pseudo_csv, _tempo, train_data_csv, limiar):
=======
def selec(conf, data_uns_ini, _pseudo_csv, CATEGORIES, _tempo, train_data_csv, limiar):
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
    """
    Selects pseudo-labels from classified unlabeled data.

    Steps performed:
    1. Rename paths in the classified data.
    2. Filter the classified data by confidence level.
<<<<<<< HEAD
    3. Select pseudo-labels and exclude them from the original unlabeled dataset.
    4. Combine the previous training set with the selected pseudo-labels.
    5. Save the new training and unlabeled datasets.

    Parameters:
        conf (dict): Configuration settings for selection.
        data_uns_ini (DataFrame): Initial classified unlabeled data.
        _pseudo_csv (str): Path to save pseudo-labels.
        _tempo (int): Current iteration time.
        train_data_csv (DataFrame): Previous training dataset.
        limiar (float): Confidence threshold for selection.

    Returns:
        dict: Sizes of datasets and paths of new training set, or False if no labels selected.
    """
    
    # Step 1: Rename paths
    
    utils.renomear_path(conf, data_uns_ini)
    print("Initial Data Preview:", data_uns_ini.head())

    # Step 2: Filter by confidence level
    data_uns_fil = data_uns_ini[data_uns_ini['confidence'] > limiar]
    print(f'Filtered data size: {len(data_uns_fil)}')

    if data_uns_fil.empty:
        print('No pseudo-labels passed the confidence filter.')
        return False

    # Step 3: Exclude selected labels from the unlabeled dataset
    data_uns_ini = data_uns_ini[~data_uns_ini['file'].isin(data_uns_fil['file'])]
    print(f'Remaining unlabeled data size: {len(data_uns_ini)}')

    # Save the remaining unlabeled data
    tempo_px = _tempo + 1
    _csv_unlabels_t = os.path.join(_pseudo_csv, f'unlabelSet_T{tempo_px}.csv')
    print(f'Saving remaining unlabeled data to {_csv_unlabels_t}')
    data_uns_ini.to_csv(_csv_unlabels_t, index=False)

    # Step 4: Combine with previous training set
    
    if _tempo == 0:
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True)
    else:
        previous_train_path = os.path.join(_pseudo_csv, f'trainSet_T{_tempo}.csv')
        train_data_csv = pd.read_csv(previous_train_path)
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True)

    # Save the new training set
    _csv_New_TrainSet = os.path.join(_pseudo_csv, f'trainSet_T{tempo_px}.csv')
    print(f'Saving new training set to {_csv_New_TrainSet}')
    New_train_data.to_csv(_csv_New_TrainSet, index=False)

    # Return summary of selections and data sizes
    return {
        'ini': len(data_uns_ini) + len(data_uns_fil),
        'select': len(data_uns_fil),
        'rest': len(data_uns_ini),
        'train': len(train_data_csv),
        'new_train': len(New_train_data),
        '_csv_New_TrainSet': _csv_New_TrainSet
    }

=======
    3. Select pseudo-labels, ensuring a minimum of 100 per class.
    4. Exclude selected labels from the original unlabeled dataset.
    5. Combine the previous training set with the selected pseudo-labels.
    6. Save the new training set.
    7. Implement stopping rules.

    Parameters:
        conf: Configuration dictionary containing settings for selection.
        data_uns_ini: Initial classified unlabeled data.
        _pseudo_csv: Path to save pseudo-labels.
        CATEGORIES: List of image classes.
        _tempo: Time variable for tracking iterations.
        train_data_csv: Previous training dataset.
        limiar: Confidence threshold for selection.

    Returns:
        Dictionary with sizes of datasets and paths of new training set, or False if no labels selected.
    """
    utils.renomear_path(conf, data_uns_ini)
    print(data_uns_ini.head())

    print("\n[select 2] Filtering by confidence")  
    _size_data_uns_ini = len(data_uns_ini)
    print(f'\nInitial size of unlabeled data: {_size_data_uns_ini}, time {_tempo} ')

    data_uns_fil = data_uns_ini.loc[data_uns_ini['confianca'] > limiar]
    print('Filtered data size:', len(data_uns_fil))

    if len(data_uns_fil) == 0:
        print('No pseudo-labels passed the confidence filter.')
        return False  # No valid selections

    print(f"\n[select 4] Excluding labels from the unlabeled dataset, time {_tempo}")
    for i in data_uns_fil['file']:
        data_uns_ini.drop(data_uns_ini[data_uns_ini['file'] == i].index, inplace=True)

    _size_uns_select = len(data_uns_fil)
    print(f'Number of selected pseudo-labels: {_size_uns_select}, time {_tempo} ')
    
    _size_uns_rest = len(data_uns_ini)
    print(f'Remaining unlabeled data: {_size_uns_rest}, time {_tempo} ')

    print(data_uns_ini.head())
    
    tempo_px = _tempo + 1
    _csv_unlabels_t = _pseudo_csv + '/unlabelSet_T' + str(tempo_px) + '.csv'
    print(f'[BUILD] Saving remaining unlabeled data at time {tempo_px}, {_csv_unlabels_t}')
    data_uns_ini.to_csv(_csv_unlabels_t, index=False, header=True)

    if _tempo == 0:
        print(f'\nSize of previous training set: {len(train_data_csv)}, time {_tempo} ')
        
        print("\n[select 5] Combining previous training set with selected pseudo-labels")
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True, sort=False)
        print(f'Size of new training set: {len(New_train_data)}, time {_tempo} ')
    else:
        _csv_TrainSet = _pseudo_csv + '/trainSet_T' + str(_tempo) + '.csv'
        print('[BUILD _csv_TrainSet', _csv_TrainSet)
        train_data_csv = pd.read_csv(_csv_TrainSet)

        print("\n[select 5] Combining previous training set with selected pseudo-labels")
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True, sort=False)

    _size_New_train_data = len(New_train_data)
    _size_train_ant = len(train_data_csv)
    print(f'\nSize of previous training set: {_size_train_ant}, time {_tempo} ')
    print(f'Size of next training set: {_size_New_train_data}, time next {tempo_px} ')
    
    # Save the new training set
    _csv_New_TrainSet = _pseudo_csv + '/trainSet_T' + str(tempo_px) + '.csv'
    print(f"\n[select 4] Saving new training set at time {tempo_px}, {_csv_New_TrainSet}")
    New_train_data.to_csv(_csv_New_TrainSet, index=False, header=True)

    tamanho = utils.bytes_to_mb(sys.getsizeof(New_train_data))
    print(f'The size of the variable {New_train_data} is approximately {tamanho} MB.')
    
    sel_0 = {
        'ini': _size_data_uns_ini, 
        'select': _size_uns_select, 
        'rest': _size_uns_rest, 
        'train': _size_train_ant, 
        'new_train': _size_New_train_data,
        '_csv_New_TrainSet': _csv_New_TrainSet
    }
    
    return sel_0
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5

if __name__ == "__main__": 
    help(classificaImgs)
    help(selec)
