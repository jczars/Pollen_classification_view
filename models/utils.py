# -*- coding: utf-8 -*-
import gc
import pandas as pd
import tensorflow as tf
import tqdm, os, sys
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.backend import get_session
import nvidia_smi
<<<<<<< HEAD
import csv
=======
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5


def create_dataSet(_path_data, _csv_data, CATEGORIES):
  """
  --> create data set in format csv
  :param: _path_data: path the dataSet
  :param: _csv_data: path with file name '.csv'
  :param: _categories: the classes the data
  :return: DataFrame with file path
  """
  data=pd.DataFrame(columns= ['file', 'labels'])
  print('_path_data ', _path_data)
  print('_csv_data ', _csv_data)
  print('CATEGORIES: ', CATEGORIES)

  c=0
  #cat_names = os.listdir(_path_data)
  for j in CATEGORIES:
      pathfile = _path_data+'/'+j
      filenames = os.listdir(pathfile)
      for i in filenames:
        #print(_path_data+'/'+j+'/'+i)
        data.loc[c] = [str(_path_data+'/'+j+'/'+i), j]
        c=c+1
  #print(c)
  data.to_csv(_csv_data, index = False, header=True)
  data_csv = pd.read_csv(_csv_data)
  print(_csv_data)
  print(data_csv.groupby('labels').count())

  return data

def create_unlabelSet(_unlabels):
  unlabelsBD=pd.DataFrame(columns= ['file'])
  c=0
  filenames = os.listdir(_unlabels)
  for i in filenames:
    #print(_unlabels+'/'+i)
    unlabelsBD.loc[c] = [str(_unlabels+'/'+i)]
    c=c+1
  print(c)
  return unlabelsBD

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


def criarTestes(_path, nmTeste):
  """
  -->create folders
  :param: _path: path the folder
  :param: nmTeste: name the test
  """
  # nome do teste: modelo+size+aug
  _dir_test = _path+'/Train_'+nmTeste+'/'
  print('folders test: ',_dir_test)
  create_folders(_dir_test, flag=0)
  return _dir_test, nmTeste

def renomear_path(conf, data_uns, verbose=0):
<<<<<<< HEAD
  unlabels_path = f"{conf['path_base']}/images_unlabels/"

  print(data_uns.head())

=======
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
  for index, row in data_uns.iterrows():
    values=row['file']
    if verbose>0:
        print('[INFO] renomear_path')
        print(values)
    isp=values.split('/')
<<<<<<< HEAD
    new_path=unlabels_path+'unlabels/'+isp[-1]
=======
    new_path=conf['unlabels']+'unlabels/'+isp[-1]
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
    data_uns.at[index,'file']=new_path
    if verbose>0:
        print(new_path)
      

def get_process_memory():
    """Return total memory used by current PID, **including** memory in shared libraries
    """
    raw = os.popen(f"pmap {os.getpid()}").read()
    # The last line of pmap output gives the total memory like
    # " total            40140K"
    memory_mb = int(raw.split("\n")[-2].split()[-1].strip("K")) // 1024
    return memory_mb

def bytes_to_mb(size_in_bytes):
    size_in_mb = size_in_bytes / (1024.0 ** 2)
    return size_in_mb

<<<<<<< HEAD

=======
def use_memo():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    use_memo=100*info.used/info.total
    if use_memo>95:
        print('para process')
        reset_keras()
        #sys.exit()
    else:
        print('continue')

def reset_keras():
    classifier=0
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    print("#"*30)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #print(tf.config.experimental.set_memory_growth('/device:GPU:0', True))
    print(tf.config.experimental.get_memory_info('GPU:0'))
    print(tf.config.experimental.reset_memory_stats('GPU:0'))

    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not install")

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted
    #print(torch.cuda.empty_cache())

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    print(set_session(tf.compat.v1.Session(config=config)))
    #set_session(tf.Session(config=config))
    print("#"*30)
    print('\n')
    
def get_menor(new_data_uns, CATEGORIES):
    lista=[]
    for cat in CATEGORIES:
        df = new_data_uns[new_data_uns['labels'] == cat]
        size=len(df)
        print(cat, size)
        lista.append(size)
    menor=min(lista)

    if menor<100:
        menor=100
    elif menor==0:
        menor =100
    print('menor', menor)

    return menor
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
def select_pseudos(pseudos, CATEGORIES, menor, _tempo):
    df_sel=pd.DataFrame(columns= ['file','labels'])

    df_cat_size=[]
    print(f"cat{'-':<17}| total")
    print("-"*30)
    for cat in CATEGORIES:
        df = pseudos[pseudos['labels'] == cat]
        if len(df)>0:
            df=df[:menor]
            size=len(df)
            print(f"{cat:<20}| {size}")
            df_cat_size.append([_tempo, cat, size])
            df_sel=pd.concat([df_sel,df])
            _size_selec=len(df_sel)
            print('Total de dados selecioandos ', _size_selec)
    return df_sel, df_cat_size
<<<<<<< HEAD
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
=======

>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5

if __name__=="__main__":
   help(create_dataSet)
   help(create_unlabelSet)
   help(create_folders)
   help(criarTestes)
<<<<<<< HEAD
   help(add_row_csv)
=======
>>>>>>> e929c9913101fa2771f1d5b3c9b817b2fe641ac5
   