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
```

To remove the project path
```bash
unset PYTHONPATH
```

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
## Description of Key Folders:

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


# Usage
## Phase 1
The project is divided into phases, following the outline of phase 1.
**Phase 1**: Separate pollen into views (Equatorial and Polar) using pseudo-labeling.

1. **Unpack the BI_5 dataset**:
   First, extract the BI_5 dataset by running the following command:

```bash
sudo tar -xzvf BD/BI_5.tar.gz -C BD/
```

2. **Perform pseudo-labeling**: To run the pseudo-labeling process, execute the following command. This will start the process based on the configuration in the specified Excel file:
```bash
python 0_pseudo_labels/pseudo_reload_train.py --path 0_pseudo_labels/Reports/config_pseudo_label_pre.xlsx --start_index 5 --end_index 1
```
in this case only one test will be performed.
```bash
python 0_pseudo_labels/pseudo_reload_train.py --path 0_pseudo_labels/Reports/config_pseudo_label_pre.xlsx --start_index 0
```
in this case all the tests configured in the spreadsheet will be executed.

3. **Create dataset**: folder containing the algorithms used to prepare the datasets.
**Cretan Pollen Dataset v1 (CPD-1)**
Folder containing the algorithms used to prepare the datasets.
Download the database available at: https://zenodo.org/records/4756361. Choose the Cropped Pollen Grains version and place it in the BD folder.
```bash
wget -P BD/ https://zenodo.org/record/4756361/files/Cropped%20Pollen%20Grains.rar?download=1 -O BD/Cropped_Pollen_Grains.rar
```
or
```bash
curl -L https://zenodo.org/record/4756361/files/Cropped%20Pollen%20Grains.rar?download=1 -o BD/Cropped_Pollen_Grains.rar
```
To extract the .rar file, you need to install the unrar tool (if not already installed):
```bash
sudo apt-get install unrar
```
This installs the unrar tool, which is necessary for extracting .rar files.
```bash
sudo unrar x BD/Cropped_Pollen_Grains.rar BD/
```

4. **Renaming Dataset Classes**
The database class names follow the syntax “1.Thymbra” (e.g., "1.Thymbra", "2.Erica"). We will rename the folders to follow a simpler format like “thymbra”.
Use the rename_folders.py script to rename the classes:

```bash
python 1_create_bd/rename_folders.py --path_data BD/Cropped\ Pollen\ Grains/
```
This command runs the rename_folders.py script to rename the class folders inside the Cropped Pollen Grains directory. Each folder name will be converted to lowercase for consistency.

5. **Resize images dataset**
This script reads the images from the Cropped Pollen Grains dataset and checks if they are in the standard size of 224 x 224. If any images do not meet these dimensions, the script creates a new dataset with all images resized to the specified size.

The resizing process uses a configuration file (config_resize.yaml) to define input and output paths, along with other parameters.

**Expected Result**:
A new dataset folder containing all images resized to 224 x 224, ensuring consistency across the dataset.

Usage: To run the resizing script with the configuration file, use the following command:

```bash
python 1_create_bd/resize_img_bd.py --config 1_create_bd/config_resize.yaml
```

6. **Separeted dataset**:
This script separates the Cropped Pollen Grains dataset into two distinct views: Equatorial and Polar. It uses a configuration file (config_separeted.yaml) to define the input and output paths, along with other processing parameters.

**Expected Results**:
At the end of the execution, the script generates reports and separates images into the specified views (Equatorial and Polar), saving them in the respective folders within the database.

**Usage**:
To run the script, ensure that the configuration file is properly set up, then execute the following command:
```bash
python python 1_create_bd/separeted_bd.py --config 1_create_bd/config_separeted.yaml
```
Ensure that the classes are correctly specified in the config_separeted.yaml file before running the script.

7. **Split the dataset into views and prepare for cross-validation**:
This script splits the dataset into separate folders to perform cross-validation.

**Inputs**:
A YAML configuration file (example: config_split.yaml) that defines the parameters for the script execution.

**Expected Outputs**:
Two generated folders: EQUATORIAL_R and POLAR_R.
Reports generated as CSV files (split_*.csv) containing information about the splits.
Example of Execution
Ensure that the config_split.yaml file is correctly configured before running the script. Then, use the following command:

```bash
python 1_create_bd/split_BD_vistas_k.py --config 1_create_bd/config_split.yaml
```

8. **Data augmentation with balancing**:

This script performs data augmentation using a balancing strategy, where the **goal** variable specifies the target number of images per class. The script counts the samples in each class, and any class below the defined goal is augmented until it reaches the target size.

**Expected Results**:  
At the end of the execution, the script generates a balanced dataset with additional images for classes that initially have fewer samples. The balanced dataset is saved in the specified output folder.

**Usage**:  
To run the script, ensure that the configuration file (`config_balanced.yaml`) is properly set up, then execute the following command:
```bash
python 1_create_bd/balanc_BD_vistas_k.py --config 1_create_bd/config_balabce.yaml 
```
## Phase 2
**Fine-tuning**:
In this phase, pre-trained models are refined to classify the datasets generated in Phase 1. The selected models include DenseNet201, MobileNet, ResNet152V2, Xception, and ResNet50. The fine-tuning process follows the DFT (Dynamic Fine-Tuning) strategy to optimize the network performance.

**Required Configuration**:
To execute the tests, a spreadsheet containing the experimental configurations is required. The default configuration file can be found in the folder:

```bash
2_fine_tuned/Reports/config_FT_vistas_121124.xlsx
```
**Execution**:
Use the following command to run the fine-tuning script:
```bash
python 2_fine_tuned/FT_DFT_K10_xlsx.py 2_fine_tuned/Reports/config_FT_vistas_121124.xlsx
```
**Failure Management**:
During the tests, especially when using memory-intensive networks like DenseNet201, failures may occur due to full memory consumption. To address this, a spreadsheet with control variables tracks the progress of the tests, allowing for recovery.

**Control Variables**:
last_test_index: Index of the last completed test.
k_fold_number: Number of the current k-fold to be executed.
num_k_folds: Number of remaining k-folds to complete the cycle.
num_tests: Number of tests remaining to be executed.
Default Configuration
```bash
last_test_index = 0
k_fold_number = 1
num_k_folds = 10
num_tests = 11
```
This configuration runs 11 tests, covering all k-folds (k=1 to k=10) for each test.

**Recovery Example**:
If a test fails at index 6 with k=9, use the following configuration to resume:
```bash
last_test_index = 6
k_fold_number = 10
num_k_folds = 1
num_tests = 1
```
This setup ensures the tests are resumed in a controlled and efficient manner.

**Expected Results**:
The results are stored in the "Reports" folder where the spreadsheet is located. The folder naming convention follows the pattern: id_test, model_name, and reports.

The output includes:

1. **CSV** files containing detailed metrics and predictions.
2. **Graphs** in JPG format, such as:
* Confusion matrix
* Training performance plot
* Boxplot of probabilities
This structure ensures organized storage and easy access to the results of each test.