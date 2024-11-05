#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:07:04 2024

@author: jczars
"""

import os
import pandas as pd

# Caminho para a base de dados
base_path = "/media/jczars/4C22F02A22F01B22/Ensembler_pollen_classification/BD/BI_5/"

# Caminhos específicos para pastas rotuladas e não rotuladas
labeled_path = os.path.join(base_path, "labels")
unlabeled_path = os.path.join(base_path, "images_unlabels/unlabels")

# Inicializar listas para guardar os dados
labeled_data = []
unlabeled_data = []

# Processar dados rotulados
for class_folder in os.listdir(labeled_path):
    class_folder_path = os.path.join(labeled_path, class_folder)
    if os.path.isdir(class_folder_path):
        for image_file in os.listdir(class_folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Formatos de imagem
                image_path = os.path.join(class_folder_path, image_file)
                labeled_data.append({"image_path": image_path, "label": class_folder})

# Processar dados não rotulados
for image_file in os.listdir(unlabeled_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Formatos de imagem
        image_path = os.path.join(unlabeled_path, image_file)
        unlabeled_data.append({"image_path": image_path})

# Salvar dados rotulados em labels.csv
labeled_df = pd.DataFrame(labeled_data)
labeled_df.to_csv("labels.csv", index=False)

# Salvar dados não rotulados em unlabels.csv
unlabeled_df = pd.DataFrame(unlabeled_data)
unlabeled_df.to_csv("unlabels.csv", index=False)

print("Arquivos CSV gerados: labels.csv e unlabels.csv")
