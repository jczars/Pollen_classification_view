import pandas as pd
from scipy.stats import wilcoxon


# Function to load the spreadsheet, perform comparisons, and apply tests
def process_spreadsheet(file_path):
    """
    Reads a spreadsheet from the 'comparar' sheet, performs specific comparisons between the datasets
    Orig 400 vs EQ 400 and Orig 400 vs PL 400, applies the Wilcoxon test, and adds interpretations of the results.
    
    :param file_path: Path to the Excel file with data.
    :return: DataFrame with comparative analyses, Wilcoxon test results, and their interpretations.
    """
    # Load the data from the 'comparar' sheet
    data = pd.read_excel(file_path, sheet_name="comparar")

    # Filter the relevant datasets for comparisons
    data_orig_eq = data[data['Base'].isin(['Orig 400', 'EQ 400'])]
    data_orig_pl = data[data['Base'].isin(['Orig 400', 'PL 400'])]

    # Specific comparisons
    data_eq_comparison = data_orig_eq[data_orig_eq['Base'].isin(['Orig 400', 'EQ 400'])]
    data_pl_comparison = data_orig_pl[data_orig_pl['Base'].isin(['Orig 400', 'PL 400'])]

    # Perform the comparison for each relevant metric
    results = []

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        # Comparison Orig 400 vs EQ 400
        try:
            comparison_eq = data_eq_comparison.pivot(index="Classes", columns="Base", values=metric).dropna()
            stat_eq, p_value_eq = wilcoxon(comparison_eq['Orig 400'], comparison_eq['EQ 400'])
            interpretation_eq = "Significant" if p_value_eq < 0.05 else "Not significant"
        except ValueError:
            stat_eq, p_value_eq, interpretation_eq = None, None, "Error"

        results.append({
            "Base_1": "Orig 400",
            "Base_2": "EQ 400",
            "Metric": metric,
            "Stat": stat_eq,
            "P-Value": p_value_eq,
            "Interpretation": interpretation_eq
        })

        # Comparison Orig 400 vs PL 400
        try:
            comparison_pl = data_pl_comparison.pivot(index="Classes", columns="Base", values=metric).dropna()
            stat_pl, p_value_pl = wilcoxon(comparison_pl['Orig 400'], comparison_pl['PL 400'])
            interpretation_pl = "Significant" if p_value_pl < 0.05 else "Not significant"
        except ValueError:
            stat_pl, p_value_pl, interpretation_pl = None, None, "Error"

        results.append({
            "Base_1": "Orig 400",
            "Base_2": "PL 400",
            "Metric": metric,
            "Stat": stat_pl,
            "P-Value": p_value_pl,
            "Interpretation": interpretation_pl
        })

    # Create a DataFrame with the test results
    results_df = pd.DataFrame(results)

    return results_df


# Path to the Excel file
file_path = "discussion/Comparar_literatura.xlsx"  # Replace with the correct path to your file

# Perform comparative analysis and Wilcoxon test
results = process_spreadsheet(file_path)

# Load the original spreadsheet
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
    # Save the Wilcoxon test results in a new sheet called "resultados"
    results.to_excel(writer, index=False, sheet_name="results")

# Display completion message
print("The results and their interpretations have been saved in the new 'resultados' sheet of the spreadsheet.")
