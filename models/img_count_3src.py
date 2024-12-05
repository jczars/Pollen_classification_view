import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_datasets_comparison(dataset_paths):
    """
    Plots a bar chart comparing the number of images across datasets (Fonte, Dataset 1, Dataset 2).
    
    Args:
        dataset_paths (dict): A dictionary with the dataset names as keys and their respective paths as values.
    """
    dataset_names = list(dataset_paths.keys())
    
    # Prepare data
    all_classes = sorted(set(
        category
        for dataset_path in dataset_paths.values()
        for category in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, category))
    ))
    
    img_count_by_dataset = {}
    
    for dataset_name, dataset_path in dataset_paths.items():
        img_count_by_dataset[dataset_name] = {}
        for category in all_classes:
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                img_count_by_dataset[dataset_name][category] = len([
                    f for f in os.listdir(category_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))
                ])
            else:
                img_count_by_dataset[dataset_name][category] = 0
    
    # Create a DataFrame for easy plotting
    df = pd.DataFrame(img_count_by_dataset).fillna(0)
    df = df.reindex(all_classes)  # Reorder classes to align across all datasets
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    
    for i, dataset_name in enumerate(dataset_names):
        ax = axes[i]
        dataset_data = df[dataset_name]
        
        # Normalize colors based on image count (darker for higher values)
        norm = plt.Normalize(vmin=dataset_data.min(), vmax=dataset_data.max())
        colors = plt.cm.Blues(norm(dataset_data))  # Use a blues colormap for better visual contrast
        
        # Create horizontal bar plot
        bars = ax.barh(df.index, dataset_data, color=colors, edgecolor='black')
        
        # Add values on the bars with appropriate color based on the value
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height() / 2, f'{int(width)}',
                    va='center', ha='left', fontsize=10, color='black' if width > 20 else 'white')
        
        ax.set_title(dataset_name, fontsize=16)
        ax.set_xlabel('Number of Images.', fontsize=12)
        ax.set_ylabel('Classes', fontsize=12)
        ax.grid(True, axis='x')
    
    # Adjust layout
    plt.tight_layout()
    

    return fig, df

# Example Usage
dataset_paths = {
    'CPD1_Cr_Rs': './BD/CPD1_Cr_Rs',
    'EQUATORIAL': './BD/CPD1_Dn_VTcr_281124/EQUATORIAL',
    'POLAR': './BD/CPD1_Dn_VTcr_281124/POLAR'
}

fig, df =plot_datasets_comparison(dataset_paths)
fig.savefig('./BD/CPD1_Dn_VTcr_281124/img_count_3src.png')
df.to_csv('./BD/CPD1_Dn_VTcr_281124/img_count_3src.csv')    
plt.show()
