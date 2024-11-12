#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:08:54 2024

@author: jczars
"""
import glob, os, sys
from tqdm import tqdm

sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)

from models import utils as utils_lib


def listar(save_dir, path_data, tipo, version, vt):
    _csv_qt_class = save_dir + '/ls_vistas_'+vt+'_qt_v'+str(version)+'.csv'
    print(_csv_qt_class)
    path_vista=path_data+'/'+vt
    cat_names = sorted(os.listdir(path_vista))
    print('\n')
    print('#'*60)
    print('Classes da vista ',vt, 'total', len(cat_names))
    print('#'*60)
    for j in tqdm(cat_names):
        pathfile = path_vista+'/'+j
        #print(pathfile)
        query=pathfile+'/*.'+tipo
        images_path = glob.glob(query)
        total=len(images_path)
    
        print(j, total)
        data = [[j,total]]
        #print(data)
        utils_lib.add_row_csv(_csv_qt_class, data)

def run(params):
    vistas=params['vistas']
    save_dir=params['save_dir']
    path_data=params['path_data']
    tipo=params['tipo']
    version=params['version']
    
    for vt in vistas:
        path_vistas=path_data+'/'+vt
        print(path_vistas)
        listar(save_dir, path_data, tipo, version, vt)

params={
        'vistas':['EQUATORIAL','POLAR'],
        'save_dir': '/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/BD/CPD1_Dn_VTcr_111124/',
        'path_data': '/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/BD/CPD1_Dn_VTcr_111124/',
        'tipo': 'png',
        'version':3  
    }   

if __name__=="__main__":
    # Sets the working directory
    os.chdir('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
    
    run(params)
    