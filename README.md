# Pollen_classification_view

This project focuses on the classification of pollen grains by taking into account their characteristic views (Equatorial, Polar). The system is divided into three phases:

Phase 1: Separate pollen into views (Equatorial and Polar) using pseudo-labeling.
Phase 2: Refine selected models and classify datasets generated in Phase 1.
Phase 3: Assemble ensembles using models from Phase 2 for more accurate classification.


## Installation

Follow the steps below to install the project.

1. Clone the repository:
```bash
git clone https://github.com/jczars/Pollen_classification_view.git

```
2. Navigate to the directory:
```bash
cd Pollen_classification_view
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Adjust Python Path (if needed):
This step is required if you face issues with the module imports.
To include the project path:
```bash

export PYTHONPATH=/media/jczars/4C22F02A22F01B22/Pollen_classification_view/:$PYTHONPATH

To remove the project path
```bash
unset PYTHONPATH

# Project Folder Structure
Below is the folder structure for the `Pollen_classification_view` project:
```bash
Pollen_classification_view
├── 0_pseudo_labels
│   ├── Reports
│   │   └── config_pseudo_label_pre.xlsx
│   ├── main_pseudo.py
├── 1_create_bd
│   ├── balanced_BD_vistas_k.py
│   ├── config_balanced.yaml
│   ├── config_resize.yaml
│   ├── config_separeted.yaml
│   ├── config_split.yaml
│   ├── rename_folders.py
│   ├── resize_img_bd.py
│   ├── separeted_bd.py
│   ├── split_BD_vistas_k.py
├── 2_fine_tuned
│   ├── Reports
│       ├── 0_DenseNet201
│       ├── 0_Test_reports_101124.xlsx
│   ├── FT_DFT_K10_Aug_xlsx.py
│   ├── FT_DFT_K10_xlsx.py
├── 3_ensemble
│   ├── Reports
│       ├── EQUATORIAL_ens_111124_reports
│       ├── POLAR_ens_111124_reports
│   ├── conf_vote.yaml
│   ├── config_eq.yaml
│   ├── config_pl.yaml
│   ├── Ensemble.py
├── BD
│   ├── BI_5
│     ├── images_unlabels
│     │   └── unlabeled
│     └── labels
│       ├── equatorial_alongada
│       ├── equatorial_circular
│       ├── equatorial_eliptica
│       ├── polar_circular
│       ├── polar_triangular
│       └── polar_tricircular
│   ├── CPD1_Dn_VTcr_111124
│       ├── EQUATORIAL
│       │    ├── castanea
│       │    ├── ceratonia
│       │    ├── ebenus
│       │    ├── ferula
│       │    ├── myrtus
│       │    ├── olea
│       │    ├── ...
│       ├── EQUATORIAL_R
│       │    ├── csv
│       │    ├── Test
│       │    │    └── k1
│       │    │        ├── castanea
│       │    │        ├── ceratonia
│       │    │        ├── ebenus
│       │    │        ├── ...
│       │    │    ├── k2
│       │    │    ├── k3
│       │    │    ├── ...
│       │    ├── Train
│       │    │    └── k1
│       │    │        ├── castanea
│       │    │        ├── ceratonia
│       │    │        ├── ebenus
│       │    │        ├── ...
│       │    │    ├── k2
│       │    │    ├── k3
│       │    │    ├── ...
│       ├── POLAR
│       │    ├── asphodelus
│       │    ├── calicotome
│       │    ├── ceratonia
│       │    ├── erica
│       │    ├── eucalyptus
│       │    ├── ferula
│       │    ├── ...
│       ├── POLAR_R
│       │    ├── csv
│       │    ├── Test
│       │    │    └── k1
│       │    │        ├── asphodelus
│       │    │        ├── calicotome
│       │    │        ├── erica
│       │    │        ├── ...
│       │    │    ├── k2
│       │    │    ├── k3
│       │    │    ├── ...
│       │    ├── Train
│       │    │    └── k1
│       │    │        ├── asphodelus
│       │    │        ├── calicotome
│       │    │        ├── erica
│       │    │        ├── ...
│       │    │    ├── k2
│       │    │    ├── k3
│       │    │    ├── ...
│   ├── CPD1_Is_Rc
│       ├── asphodelus
│       ├── calicotome
│       ├── castanea
│       ├── ceratonia
│       ├── ...
│   ├── Cropped Pollen Grains
│       ├── 1.Thymbra
│       ├── 2.Erica
│       ├── 3.Castanea
│       ├── 4.Eucalyptus
│       ├── ...
├── modulos
│   ├── del_folders_limiar.py
│   ├── get_calssifica.py
│   ├── get_data.py
│   ├── listar vistas.py
│   ├── maneger_gpu.py
│   ├── models_pre.py
│   ├── models_train.py
│   ├── reports_build.py
│   ├── reports_ens.py
│   ├── sound_test_finalizado.py
│   ├── utils.py
│   ├── voto_majoritary.py

```
### Description of Key Folders:

- **`0_pseudo_labels/`**: Contains scripts and reports for the pseudo-labeling process.
  - **`Reports/`**: Reports generated during the pseudo-labeling process.
  - **`pseudo_reload_train.py`**: Main script used to generate pseudo-labels.

- **`1_create_bd/`**: Scripts for creating balanced and resized datasets.
  - **`balanced_BD_views.py`**: Script to create balanced datasets for equatorial and polar views.
  - **`config_balanced.yaml`**: Configuration file for balancing datasets.
  - **`config_resize.yaml`**: Configuration file for resizing images.
  - **`resize_img_bd.py`**: Script for resizing the dataset images.
  - **`split_BD_views.py`**: Script for splitting datasets based on views.

- **`2_fine_tuned/`**: Contains scripts for fine-tuning pre-trained models and generating reports.
  - **`FT_DFT_K10_Aug_xlsx.py`**: Script for fine-tuning models with data augmentation.
  - **`FT_DFT_K10_xlsx.py`**: Script for fine-tuning models without data augmentation.

- **`3_ensemble/`**: Contains scripts for creating ensemble models using trained networks.
  - **`Ensemble.py`**: Main script for generating ensemble models.
  - **`conf_vote.yaml`**: Configuration file for ensemble voting strategies.

- **`modules/`**: Contains various utility scripts and modules for data handling, model training, and GPU management.
  - **`get_classifica.py`**: Script responsible for classification.
  - **`get_data.py`**: Script for data loading and preprocessing.
  - **`manage_gpu.py`**: Script for managing GPU resources.
  - **`models_pre.py`**: Models used for data preprocessing.
  - **`models_train.py`**: Models for training neural networks.
  - **`reports_build.py`**: Script for generating reports.
  - **`utils.py`**: Utility functions for various tasks.

- **`BD/`**: Database folder containing labeled and unlabeled pollen grain images.
  - **`BI_5/`**: The primary dataset with raw images and labels.
  - **`images_unlabels/`**: Contains unlabeled images for classification.
    - **`unlabeled/`**: Folder with unlabeled images for pseudo-labeling.
  - **`labels/`**: Classification labels categorized by classes.
    - Each folder inside **`labels/`** contains images for specific classes (e.g., `equatorial_circular`, `polar_triangular`, etc.).


## Usage

Follow these steps to use the project:

1. **Unpack the BI_5 dataset**:
   First, extract the BI_5 dataset by running the following command:

```bash
   sudo tar -xzvf BD/BI_5.tar.gz
´´´

2. **Perform pseudo-labeling**: To run the pseudo-labeling process, execute the following command. This will start the process based on the configuration in the specified Excel file:
```bash
  python 0_pseudo_labels/pseudo_reload_train.py --path 0_pseudo_labels/Reports/config_pseudo_label_pre.xlsx --start_index 5 --end_index 1




