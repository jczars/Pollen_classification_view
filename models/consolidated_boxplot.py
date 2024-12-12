import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report

def consolidator(folder, prefix, k):
    """
    Consolidates multiple CSV files into a single DataFrame.

    Parameters:
        folder (str): Directory where the CSV files are located.
        prefix (str): File name prefix used to identify the files.
        k (int): Total number of CSV files to consolidate.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all CSV files.
    """
    #Test_0_0_DenseNet201_df_correct_k1.csv

    combined_df = []
    for i in range(1, k + 1):
        file_path = os.path.join(folder, f"{prefix}{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                combined_df.append(df)
                print(f"File {file_path} loaded with {len(df)} rows.")
            else:
                print(f"File {file_path} is empty. Skipping.")
        else:
            print(f"File {file_path} not found. Skipping.")
    
    if not combined_df:
        raise ValueError("No valid CSV files to consolidate.")
    
    consolidated_df = pd.concat(combined_df, ignore_index=True)
    print(f"Total rows in consolidated DataFrame: {len(consolidated_df)}")
    return consolidated_df


def plot_confidence_boxplot(df_correct, type='correct'):

    if type == 'correct':
        title="Consolidated Confidence Scores for Correct Classifications"
    else:
        title="Consolidated Confidence Scores for Incorrect Classifications"
    fig=plt.figure(figsize=(9, 6), dpi=100)

    sns.set_style("whitegrid")

    # Adicionando Título ao gráfico
    sns.boxplot(y=df_correct["true_label"], x=df_correct["confidence"])
    plt.title(title, loc="center", fontsize=18)
    plt.xlabel("Confidence")
    plt.ylabel("classes")

    return fig

def saved(folder):
    # Automate the output path
    save_dir = os.path.join(
        os.path.dirname(folder.rstrip('/')),  # Remove the last slash and get the parent directory
        os.path.basename(folder.rstrip('/')) + '_consolidated/'  # Add '_aggregated' to the folder name
    )

    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print(f"Automated save directory: {save_dir}")
    return save_dir  # Return the directory for future use
def extract_test_info(folder_path):
    """
    Extracts the test id and model name from the folder path.

    Parameters:
        folder_path (str): Path to the folder containing the test name.

    Returns:
        tuple: test id and model name as strings.
    """
    folder_name = os.path.basename(folder_path.rstrip('/'))  # Remove the final slash and get the folder name
    parts = folder_name.split('_', 1)  # Split the folder name at the first occurrence of "_"
    
    # Ensure the folder name is correctly structured and split
    if len(parts) >= 2:
        test_id = parts[0]  # The first part is the test_id
        model_name = parts[1].split('_')[0]  # The second part is the model name (before any subsequent '_')
        return test_id, model_name
    else:
        raise ValueError(f"The folder name '{folder_name}' does not match the expected format.")
    
def run(folder, k=10):
   test_id, model_name = extract_test_info(folder)
   prefix_correct=f"Test_{test_id}_{test_id}_{model_name}_df_correct_k"
   prefix_incorrect=f"Test_{test_id}_{test_id}_{model_name}_df_incorrect_k"
   print(prefix_correct)

   df_correct=consolidator(folder, prefix_correct, k)
   df_incorrect=consolidator(folder, prefix_incorrect, k)
   
   fig_correct=plot_confidence_boxplot(df_correct, type='correct')
   fig_incorrect=plot_confidence_boxplot(df_incorrect, type='incorrect')

   save_dir = saved(folder)
   boxplot_correct_image = os.path.join(save_dir, 'consolidated_boxplot_correct.png')
   boxplot_incorrect_image = os.path.join(save_dir, 'consolidated_boxplot_incorrect.png')

   fig_correct.savefig(boxplot_correct_image)
   fig_incorrect.savefig(boxplot_incorrect_image)

   df_correct.to_csv(os.path.join(save_dir, 'consolidated_df_correct.csv'), index=False)
   df_incorrect.to_csv(os.path.join(save_dir, 'consolidated_df_incorrect.csv'), index=False)    


if __name__ == "__main__":
    folder = './results/phase2/reports_cr_13_400/1_DenseNet201_reports/'    
    k=10
    run(folder, k)