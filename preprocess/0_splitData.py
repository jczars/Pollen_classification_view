# -*- coding: utf-8 -*-

# Funções
import os
from math import ceil
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil, glob
#from imgaug import augmenters as iaa

#split data
from sklearn.model_selection import StratifiedKFold

def create_dataSet(_path_data, _csv_data, _categories):
  """
  --> create data set in format csv
  :param: _path_data: path the data
  :param: _csv_data: path with file name .csv
  :param: _categories: the classes the data
  :return: DataFrame with file path
  """
  data=pd.DataFrame(columns= ['file', 'labels'])

  c=0
  #cat_names = os.listdir(_path_data)
  for j in tqdm(_categories):
      pathfile = _path_data+'/'+j
      filenames = os.listdir(pathfile)
      for i in filenames:
        #print(_path_data+'/'+j+'/'+i)
        data.loc[c] = [str(_path_data+'/'+j+'/'+i), j]
        c=c+1
  #print(c)
  data.to_csv(_csv_data, index = False, header=True)
  data_csv = pd.read_csv(_csv_data)
  print('\n path do csv_data: ',_csv_data)
  print(data_csv.groupby('labels').count())
  return data

def create_folders(_save_dir, flag=1):
  """
  -->create folders
  :param: _save_dir: path the folder
  :param: flag: rewrite the folder, (1 for not and display error: 'the folder already exists)
  """
  if os.path.isdir(_save_dir):
    if flag:
      raise FileNotFoundError("folders test already exists: ", _save_dir)
    else:
      print('folders test already exists: ', _save_dir)
  else:
      os.mkdir(_save_dir)
      print('create folders test: ', _save_dir)

def kfold_split(data_csv, _path, _k, base):
  Y = data_csv[['labels']]
  n=len(Y)
  print('Total de dados ',n)
  kfold = StratifiedKFold(n_splits=int(_k), random_state=7, shuffle=True)

  # 1- divisão train, test
  k=1
  for production_index, test_index in kfold.split(np.zeros(n),Y):
    prod = data_csv.iloc[production_index]
    test = data_csv.iloc[test_index]



    print('total_train_index ',len(production_index))
    print('total_test_index ',len(test_index))
    print(k)

    #Salvar split data
    _csv_train=_path+'/'+base+'_trainSet_k'+str(k)+'.csv'
    _csv_test=_path+'/'+base+'_testSet_k'+str(k)+'.csv'

    prod.to_csv(_csv_train, index = False, header=True)
    test.to_csv(_csv_test, index = False, header=True)

    train = pd.read_csv(_csv_train)
    #print(train)
    print(train.groupby('labels').count())
    k+=1

"""# copy images"""

def copy_imgs(training_data, dst):
    #print('dst0-->', dst)
    for i in training_data['file']:
        #print(i)
        pth1=i.split('/')[0]
        #print('src----_> ',pth1)
        folder=i.split('/')[-2]
        print('folder--> ',folder)
        inst=i.split('/')[-1]
        print('inst----> ',inst)

        dst_new=dst+'/'+folder+'/'
        print('dst_new--> ', dst_new)
        create_folders(dst_new, flag=0)

        file_new= dst_new+inst
        print('file_new--> ', file_new)
        #copiar arquivo
        shutil.copy(i, file_new)

"""## build"""

def copy_img_k(_path_csv, PATH_TRAIN, NM_BASE, _k, tipo='train'):
  for i in range(_k):
    k=i+1
    folder=PATH_TRAIN+'/k'+str(k)
    create_folders(folder, flag=0)
    print('k', folder)

    path_csv=_path_csv+NM_BASE+'_'+tipo+'Set'+'_k'+str(k)+'.csv'
    print('path_csv ',path_csv)

    data_imgs=pd.read_csv(path_csv)
    copy_imgs(data_imgs, folder)

"""# quantizar"""

def quantizar(dst, CATEGORIES, _k):
  for i in range(_k):
    k=i+1
    folder=dst+'/k'+str(i+1)
    print('k', folder)

    for cat in CATEGORIES:
      path =folder+'/'+cat+'/*.png'
      images_path = glob.glob(path)
      print(cat, len(images_path))

def copy_csv(src, dst_csv):
  query_csv=src+"/*.csv"
  images_path = glob.glob(query_csv)
  print(images_path)

  for i in images_path:
    folder=i.split('/')
    inst =folder[-1]
    file_new= dst_csv+'/'+inst
    print(file_new)
    shutil.copy(i, file_new)

"""# Esturtura das pastas
* Train
* Test
* csv
"""

def estrutura(PATH_BD):
  """
  -->Create structure folders:
  Exemple:
    PATH_BD='/content/drive/MyDrive/CPD1_EQUATORIAL_Bal0'
    \PATH_BD
      \Train
        \k1
        \k2
        \k3
        ...
      \Test
        \k1
        \k2
        \k3
        ...
      \csv
  """

  create_folders(PATH_BD, flag=0)
  train=PATH_BD+'/Train'
  create_folders(train, flag=0)
  test=PATH_BD+'/Test'
  create_folders(test, flag=0)
  csv=PATH_BD+'/csv'
  create_folders(csv, flag=0)

"""#Main"""

def copy_base(BD_SRC, BD_NEW, NM_BASE, CATEGORIES, _K):
  # fase 1: criar a base e copiar as imagens para a pasta treino
  print('Criando estrutura de pastas...')
  estrutura(BD_NEW)

  print('criando dataSet csv...')
  PATH_CSV=BD_NEW+'/csv/'
  _csv_data=PATH_CSV+NM_BASE+'.csv'
  data_csv=create_dataSet(BD_SRC, _csv_data, CATEGORIES)

  print('criando os kfolds...')
  kfold_split(data_csv, PATH_CSV, _K, NM_BASE)

  print('copiando as imagens para BD\Train')
  PATH_TRAIN=BD_NEW+'/Train/'

  copy_img_k(PATH_CSV, PATH_TRAIN, NM_BASE, _K, tipo='train')
  quantizar(PATH_TRAIN, CATEGORIES, _K)

  print('copiando as imagens para BD\Test')
  PATH_TEST=BD_NEW+'/Test/'

  copy_img_k(PATH_CSV, PATH_TEST, NM_BASE, _K, tipo='test')
  quantizar(PATH_TEST, CATEGORIES, _K)

def run(BD_SRC, BD_DST, NM_BASE, _K):
    CATEGORIES = sorted(os.listdir(BD_SRC))
    copy_base(BD_SRC, BD_DST, NM_BASE, CATEGORIES, _K)

if __name__=="__main__":
    NM_BASE = 'BD_CPD1'
    BD_SRC = '/media/jczars/4C22F02A22F01B22/Pollen_classification_view/BD/CPD1_Cr_Rs/'
    BD_DST = '/media/jczars/4C22F02A22F01B22/Pollen_classification_view/BD/CPD1_Cr_Rs_500/'
    _K=10   
    print('BD_Split ', BD_SRC)
    
    run(BD_SRC, BD_DST, NM_BASE,_K)
