import numpy as np
import pandas as pd
import yaml, os, sys
import argparse

from sklearn.metrics import accuracy_score
os.environ["tf_gpu_allocator"]="cuda_malloc_async"
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)

from models import reports_build as reports
from models import  utils, reports_ens

def majority_vote(predictions, probabilities):
    """
    Perform majority voting on a list of predictions, with tie-breaking rules.

    Parameters
    ----------
    predictions : list of str
        A list containing the predicted class labels from different models.
    probabilities : list of float
        A list containing the probabilities associated with the predictions.

    Returns
    -------
    str
        The predicted class label based on majority vote.
    """
    # Count the frequency of each class label
    unique_labels, counts = np.unique(predictions, return_counts=True)
    
    if len(unique_labels) == len(predictions):  # Case where each model votes differently
        # In case of a tie, choose the prediction with the highest probability
        max_prob_index = np.argmax(probabilities)
        return predictions[max_prob_index]
    else:
        # Case of majority voting
        max_vote_label = unique_labels[np.argmax(counts)]
        return max_vote_label

def read_and_filter_predictions(file_path: str, sheet_names: list) -> pd.DataFrame:
    """
    Read predictions from multiple sheets in an Excel file, combining the data.

    Parameters
    ----------
    file_path : str
        Path to the Excel file.
    sheet_names : list
        List of sheet names where predictions are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined predictions from all sheets.
    """
    combined_data = []

    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df['model'] = sheet  # Add the model name as a column
        combined_data.append(df[['k', 'Index', 'Filename', 'y_true', 'y_pred', 'Probability', 'model']])

    # Combine all predictions from the sheets
    return pd.concat(combined_data, ignore_index=True)

def apply_majority_voting(filtered_df: pd.DataFrame, model_names: list) -> dict:
    """
    Apply majority voting for each value of `k` available in the DataFrame.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        DataFrame containing all predictions from the sheets.
    model_names : list
        List of model names.

    Returns
    -------
    dict
        A dictionary where each key is a value of `k` and the value is a DataFrame with the final voting results.
    """
    final_results_by_k = {}

    # Identify all unique values of `k`
    unique_k_values = filtered_df['k'].unique()

    for k_value in unique_k_values:
        k_filtered_df = filtered_df[filtered_df['k'] == k_value]
        final_results = []

        for idx, group in k_filtered_df.groupby(['Index', 'Filename', 'y_true']):
            predictions = group['y_pred'].values
            probabilities = group['Probability'].values

            elected = majority_vote(predictions, probabilities)

            # Define the status based on the comparison between y_true and elected
            sit = 'Correct' if elected == group['y_true'].iloc[0] else 'Incorrect'

            # Store the result with model names instead of m1, m2, m3
            result_entry = {
                "Index": idx[0],
                "Filename": idx[1],
                "y_true": idx[2],
                "elected": elected,
                "probability": max(probabilities),
                "status": sit
            }

            # Add the predictions of each model with their respective names
            for model_name in model_names:
                if model_name in group['model'].values:
                    result_entry[model_name] = group.loc[group['model'] == model_name, 'y_pred'].values[0]

            final_results.append(result_entry)

        # Store the results DataFrame for the current value of `k`
        final_results_by_k[f'k{k_value}'] = pd.DataFrame(final_results)     

    return final_results_by_k

def create_performance_summary(final_results_by_k: dict) -> pd.DataFrame:
    """
    Create a performance summary for each value of `k`.

    Parameters
    ----------
    final_results_by_k : dict
        A dictionary where each key is a value of `k` and the value is a DataFrame with the final voting results.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the performance for each `k`.
    """
    summary_data = []

    for k_value, results_df in final_results_by_k.items():
        correct_count = (results_df['status'] == 'Correct').sum()
        incorrect_count = (results_df['status'] == 'Incorrect').sum()
        total_count = correct_count + incorrect_count
        performance_percentage = (correct_count / total_count * 100) if total_count > 0 else 0
        
        summary_data.append({
            "k": k_value,
            "Correct": correct_count,
            "Incorrect": incorrect_count,
            "Performance (%)": performance_percentage
        })

    return pd.DataFrame(summary_data)

def reports_gen(y_true, y_pred, categories, df_correct, nm_model, k, save_dir=None):
    
    # Confusion matrix
    matrix_fig, mat = reports.plot_confusion_matrixV4(y_true, y_pred, categories, normalize=None)
    df_mat = pd.DataFrame(mat, index=categories, columns=categories)
    
    # Boxplot, classification report and training metrics
    boxplot_fig = reports.plot_confidence_boxplot(df_correct)
    class_report = reports.generate_classification_report(y_true, y_pred, categories)
        
    # Save files if directory is specified
    if save_dir:
        folder_name=f"{nm_model}_reports/"
        save_dir = os.path.join(save_dir, folder_name)
        print(save_dir)
        utils.create_folders(save_dir, flag=0)
        print("save graph")
        df_mat.to_csv(f'{save_dir}/Test_{nm_model}_mat_conf_k{k}.csv')
        df_correct.to_csv(f'{save_dir}/Test_{nm_model}_df_correct_k{k}.csv')
        
        class_report.to_csv(f'{save_dir}/Test_{nm_model}_Class_reports_k{k}.csv')
                
        matrix_fig.savefig(f'{save_dir}/Test_{nm_model}_mat_conf_k{k}.jpg')
        boxplot_fig.savefig(f'{save_dir}/Test_{nm_model}_boxplot_k{k}.jpg')
    
    # Calculate metrics
    me = reports.calculate_metrics(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Return evaluation metrics
    me = {
        'test_accuracy': accuracy,
        'precision': me['precision'],
        'recall': me['recall'],
        'fscore': me['fscore'],
        'kappa': me['kappa'],
    }
    
    return me

def run(config):
    excel_file = config['excel_file']
    model_names = config['model_names']

    filtered_df = read_and_filter_predictions(excel_file, model_names)
    final_results_by_k = apply_majority_voting(filtered_df, model_names)
    performance_summary_df = create_performance_summary(final_results_by_k)

    # Save the final results in separate sheets for each `k` and the summary
    with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl') as writer:
        for k_value, final_df in final_results_by_k.items():
            final_df.to_excel(writer, sheet_name=k_value, index=False)            
        performance_summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print("Final voting and performance summary saved successfully.")

    reports_ens.run(excel_file)


# Load configuration from YAML file
def load_config(config_file: str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example of usage
if __name__ == "__main__":
    default_yaml = '/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/3_ensemble/conf_vote.yaml'

    # Argument parser to accept configuration file as parameter
    parser = argparse.ArgumentParser(description='Process model predictions using a configuration file.')
    parser.add_argument('config', nargs='?', default=default_yaml, type=str, help='Path to the YAML configuration file (default: config.yaml)')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level (default is 0)')
    
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    run(config)