# Pollen_classification_view

Pollen classification taking into account the characteristic view of the pollen. This system is phased, where the first is responsible for separating the pollen into views [Equatorial, Polar]. The second phase is responsible for refining the selected networks and classifying the bases generated in phase 1. The third phase is responsible for assembling committees using the networks from phase 2.

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
# Project Folder Structure
```bash
Below is the folder structure for the `Pollen_classification_view` project:

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
### Folder Description:
- **0_pseudo_labels/**: Contains scripts and reports for the pseudo-labeling process.
  - **Reports/**: Reports generated during the pseudo-labeling process.
  - **main_pseudo.py**: Main script used to generate pseudo-labels.

- **modulos/**: Contains the modules and scripts used in the project, organized by functionality.
  - **get_calssifica.py**: Script responsible for classification.
  - **get_data.py**: Script for data acquisition and manipulation.
  - **maneger_gpu.py**: Script for managing GPU resources.
  - **models_pre.py**: Models used for data preprocessing.
  - **models_train.py**: Models for training neural networks.
  - **reports_build.py**: Script for report generation.
  - **utils.py**: Utility functions for the project.

- **BD/**: Contains the database with images and labels.
  - **BI_5**: The `BI_5` dataset.
  - **images_unlabels/**: Unlabeled images for classification.
    - **unlabeled**: Folder containing unlabeled images.
  - **labels/**: Classification labels.
    - Each folder inside **labels/** contains label files for a specific class.

This structure can be included in your `README.md` to provide a clear overview of the project's organization and make it easier to navigate.



## Usage

Here are some examples of how to use the project:

```bash
npm start
