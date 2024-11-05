import os
import openpyxl
import tensorflow as tf
import pandas as pd
import sys
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)

from models import steps
from models import get_data
from models import utils

#from Library import send_whatsApp_msn

# Set the environment variable to reduce the log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL

# For other libraries (e.g. Keras) you may need to adjust the logging configuration as well
tf.get_logger().setLevel('ERROR')  # To ensure that TensorFlow does not print informational log messages

# Sets the working directory
os.chdir('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/0_pseudo_labels/')

# Verifica o diretório atual
print("Diretório de trabalho atual:", os.getcwd())

def rel_data(_tempo, me, return_2, rel_book, nm_csv, idt):
  "--> Salva os dados na planilha"
  
  print("\n[INFO] nome da planilha ",nm_csv)
  rel_book = openpyxl.load_workbook(nm_csv)

  print(rel_book.sheetnames) # visualiza as abas existentes
  rel_book.save(nm_csv) # salva a planilha
  
  
  _teste='Tabela'
  if _teste in rel_book.sheetnames:
      print('exists')
  else:
      rel_book.create_sheet(_teste) # cria uma aba
  Met_page=rel_book[_teste] # inserir dados
  if _tempo==0:
      print('[INFO] -rel_data- salvando o cabeçalho do teste')
      cols_exe=['Tempo', 'test_loss', 'test_accuracy',
                       'precisio', 'recall', 'fscore', 'kappa',
                       'str_time', 'end_time', 'delay', 'epoch_finish',
                       'ini', 'select', 'rest', 'train','new_train','id_test']
      Met_page.append(cols_exe) # primeira linha

  data=[str(_tempo),
        me['test_loss'],
        me['test_accuracy'],
        me['precision'],
        me['recall'],
        me['fscore'],
        me['kappa'],
        me['str_time'],
        me['end_time'],
        me['delay'],
        me['num_epoch'],
        return_2['ini'],
        return_2['select'],
        return_2['rest'],
        return_2['train'],
        return_2['new_train'],
        idt
        ]

  Met_page.append(data)
  rel_book.save(nm_csv)

def rel_select(rel_book, nm_csv, _tempo, df_cat_size, idt):
    print("\n[INFO] nome da planilha ",nm_csv)
    rel_book = openpyxl.load_workbook(nm_csv)
    print(rel_book.sheetnames) # visualiza as abas existentes
    _sheet_name='Select'
    if _sheet_name in rel_book.sheetnames:
        print('exists')
    else:
        rel_book.create_sheet(_sheet_name) # create sheet
        
    sheet_active=rel_book[_sheet_name] # sheet_active
    if _tempo==0:
        print('[INFO] -rel_data- salvando o cabeçalho do teste')
        cols_exe=['id_test','time', 'labels', 'size']
        sheet_active.append(cols_exe) # primeira linha
    
    print('[BUILD] df_cat_size')
    print(df_cat_size)

    
    for row in (df_cat_size):
        new_row=idt,row[0],row[1],row[0]
        sheet_active.append(new_row) #insert data

    #sheet_active.append(df_cat_size) #insert data
    rel_book.save(nm_csv) # salva a planilha

def run(nm_csv, ls):
  
  print("\n[INFO] nome da planilha ",nm_csv) 
  rel_book = openpyxl.load_workbook(nm_csv)
  print(rel_book.sheetnames) # visualiza as abas existentes
  rows = pd.read_excel(nm_csv, sheet_name="Sheet")
  
  for idx in range(len(rows)):
    utils.reset_keras()
    idt=ls+idx
    conf=rows.loc[idt]
    print('conf ', conf)
    
    _tempo=0

    _unlabels_zero=True
    #_labels=str(conf['labels'])
    _labels=conf['labels']
    print(_labels)
    #/media/jczars/4C22F02A22F01B22/01_PROP_TESE/01_PSEUDO_BD/BI_5/labels/
    CATEGORIES = sorted(os.listdir(_labels))
    print('\nrun.step_0')
    return_0 =steps.step_0(conf, _labels, CATEGORIES)
    
    print('\nrun.step_1')
    model_inst, me, =steps.step_1(conf, CATEGORIES, return_0, _tempo)

    while _unlabels_zero:
      utils.reset_keras()
      print('\nrun.step_2')
      print('[buil] return_0 ', return_0)
      """
      4-seleção de novos pseudos
      """
      returns_2=steps.step_2(conf, 
                             _tempo,  
                             str(conf['unlabels']),                                           
                             model_inst, 
                             CATEGORIES,                                         
                             return_0,conf['limiar'])

      #rel tempo 0
      #del return_0
      #gc.collect()
      
      if returns_2==0:
          _unlabels_zero=False
          pass
      else:
          rel_data(_tempo, me, returns_2, rel_book, nm_csv, idt)
          #rel_select(rel_book, nm_csv, _tempo, df_cat_size, idt)
          
    
          print(_unlabels_zero)
          _tempo=_tempo+1
          print('################### Tempo #################### ')
          print(_tempo)
          """
          -ler new_train_data
          -split New_train_data in train, val
          -salvar os dados na tabela de teste
          """
    
          train, val=get_data.reload_data_train(conf,returns_2['_csv_New_data'])
          model_inst, me=steps.step_3(conf, CATEGORIES, train, val,_tempo, model_inst, returns_2)
    print(f'Teste finalizado id_teste {idt}')
    del returns_2
    del train, val
    del model_inst, me

if __name__=="__main__":
  nm_csv='/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/0_pseudo_labeling/REPORTS/0-reports_pseudo_label_pre.xlsx'
  ls=10
  run(nm_csv, ls)
  