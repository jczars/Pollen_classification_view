#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, shutil, glob
from tqdm import tqdm

def del_folder(path, flag=0):
    if os.path.isdir(path):
        print('O path exists ',path)
        if flag==1:
            shutil.rmtree(path)
    else:
        print('path not found')


# In[2]:
def del_vistas(params):
    for vt in params['vistas']:
        path_vistas=params['bd_src']+'/'+vt
        #print(path_vistas)
        cat_names = sorted(os.listdir(path_vistas))
        
        for j in tqdm(cat_names):
            path_folder = path_vistas+'/'+j
            print(path_folder)
            query=path_folder+params['tipo']
            images_path = glob.glob(query)
            total=len(images_path)
            print(total)
            if total< params['limiar']:
                print('del folders')
                del_folder(path_folder, params['flag'])
        #break


# In[3]:

"""# Main"""
# Sets the working directory
os.chdir('/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/')
params={'tipo':'/*.png',
        'bd_src': "/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/BD/CPD1_Dn_VTcr_111124/",
        'vistas':['EQUATORIAL','POLAR'],
        'flag': 1, #0-não deleta, 1-deleta
        'limiar': 20 # menor quantidade de exemplos por classes
        }



# In[ ]:
if __name__=="__main__":
    del_vistas(params)


