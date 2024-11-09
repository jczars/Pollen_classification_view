# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

def aug_param(_aug):
    #Augmentation
    print('\n ######## Data Generator ################')
    #https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    if _aug=='sem':
        aug=dict(rescale=1./255)
    if _aug== 'aug0':
        aug=dict(rescale=1./255,
                 brightness_range=[0.2,0.8],
                 horizontal_flip = True)

    idg = ImageDataGenerator(**aug)
    return idg

def load_data_train(training_data, val_data, aug, input_size):
  """
    -->loading train data
    :param: training_data: ptah at dataset
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
  idg = aug_param(aug)

  train_data_generator = idg.flow_from_dataframe(training_data,
                                            x_col = "file",
                                            y_col = "labels",
                                            target_size=input_size,
                                            class_mode = "categorical",
                                            shuffle = True)
  valid_data_generator = idg.flow_from_dataframe(val_data,
                                                x_col = "file",
                                                y_col = "labels",
                                                target_size=input_size,
                                                class_mode = "categorical",
                                                shuffle = True)
  return train_data_generator, valid_data_generator

def reload_data_train(conf, _csv_training_data, SPLIT_VALID=0.2):
    """
    -->loading train data
    :param: training_data: ptah at dataset
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
    print('training_data_path ',_csv_training_data)
    training_data=pd.read_csv(_csv_training_data)
    idg = aug_param(conf['aug'])

    idg = ImageDataGenerator(rescale=1. / 255, validation_split=SPLIT_VALID)
    img_size=conf['img_size']
    input_size=(img_size, img_size)

    train_generator = idg.flow_from_dataframe(
        dataframe=training_data,
        x_col = "file",
        y_col = "labels",
        target_size=input_size,
        class_mode = "categorical",
        shuffle = True,
        subset='training')

    val_generator = idg.flow_from_dataframe(
        dataframe=training_data,
        x_col = "file",
        y_col = "labels",
        target_size=input_size,
        class_mode = "categorical",
        shuffle = True,
        subset='validation')

    return train_generator, val_generator

def load_data_test(test_data, input_size):
  idg = ImageDataGenerator(rescale=1. / 255)
  test_data_generator = idg.flow_from_dataframe(test_data,
                                            x_col = "file",
                                            y_col = "labels",
                                            target_size=input_size,
                                            class_mode = "categorical",
                                            shuffle = False)
  return test_data_generator
                             
def load_unlabels(conf):
    
  idg = ImageDataGenerator(rescale=1. / 255)
  path=str(conf['unlabels'])
  print("conf['unlabels']: ", str(path))
  
  img_size=conf['img_size']
  input_size=(img_size, img_size)
  print('input_size ',(img_size, img_size))

  unalbels_generator = idg.flow_from_directory(
      directory=path,
<<<<<<< HEAD
      target_size=(input_size),
=======
      target_size=(224,224),
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
      color_mode="rgb",
      batch_size=conf['batch_size'],
      class_mode=None,
      shuffle=False,
      seed=42)
  return unalbels_generator

def splitData(data_csv, path_save, name_base):
  """
  -->Split dataSet into training and testing data
  :param: data_csv: dataSet in csv format
  :param: path_save: path to save training and testing data
  :param: name_base: name to save the data
  """

  prod_csv, test_csv = train_test_split(data_csv, test_size=0.2, shuffle=True)
  train_csv, val_csv = train_test_split(prod_csv, test_size=0.2, shuffle=True)
  #print('Train ',train_csv.shape)
  #print('Test ',test_csv.shape)

  #Salvar split data
  _path_train=path_save+'/'+name_base+'_trainSet.csv'
  _path_test=path_save+'/'+name_base+'_testSet.csv'
  _path_val=path_save+'/'+name_base+'_valSet.csv'
  train_csv.to_csv(_path_train, index = False, header=True)
  test_csv.to_csv(_path_test, index = False, header=True)
  val_csv.to_csv(_path_val, index = False, header=True)

  training_data = pd.read_csv(_path_train)
  print('\n Train split')
  print(training_data.groupby('labels').count())
  test_data = pd.read_csv(_path_test)
  print('\n Test split')
  print(test_data.groupby('labels').count())

  val_data = pd.read_csv(_path_val)
  print('\n Val split')
  print(val_data.groupby('labels').count())

  return _path_train,_path_val,_path_test

if __name__=="__main__":
   help(load_data_train)
   help(reload_data_train)
   help(load_data_test)
   help(load_unlabels)
   help(splitData)