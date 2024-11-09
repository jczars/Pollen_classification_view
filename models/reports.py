from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, CATEGORIES, nm_model,
                          save_dir='', tempo=0, verbose=0):
    """
    Plots and optionally saves a confusion matrix.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        CATEGORIES: List of image classes.
        nm_model: Name of the model.
        save_dir: Path to save the confusion matrix; default is '' (do not save).
        tempo: Time or fold variable for naming.
        verbose: If 1, print the graph of the confusion matrix; if 0, do not print.

    Returns:
        None
    """
    print('Confusion Matrix')
    mat = confusion_matrix(y_true, y_pred, normalize=None)
    
    # Save matrix as CSV
    print('Saving matrix as CSV in', save_dir)
    df_mat = pd.DataFrame(mat)

    if save_dir:
        df_mat.to_csv(save_dir + 'mat_conf_test_' + nm_model + '_tempo_' + str(tempo) + '.csv')

    my_dpi = 100
    plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi)
    ax = plt.subplot()
    sns.heatmap(mat, cmap="Blues", annot=True)  # annot=True to annotate cells

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=8)

    # Labels, title, and ticks
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    ax.xaxis.set_ticklabels(CATEGORIES)
    ax.yaxis.set_ticklabels(CATEGORIES)

    if save_dir:
        plt.savefig(save_dir + 'mat_conf_test_' + nm_model + '_tempo_' + str(tempo) + '.jpg')
    if verbose == 1:
        plt.show()

def predict_data_generator(conf, return_0, test_data_generator, model, tempo,
                            CATEGORIES, verbose=2):
    """
    Generates predictions and evaluation metrics (accuracy, precision, recall, fscore, kappa).

    Parameters:
        conf: Configuration dictionary.
        return_0: Dictionary containing save directory.
        test_data_generator: Images from Image Data Generator.
        model: Loaded model.
        tempo: Time or fold variable.
        CATEGORIES: List of image classes.
        verbose: Enable printing (0, 1, or 2).

    Returns:
        y_true: True labels.
        y_pred: Predicted labels.
        df_cor: DataFrame with correct predictions.
    """
    batch_size = conf['batch_size']
    nm_model = conf['model']
    save_dir = return_0['save_dir_train']

    filenames = test_data_generator.filenames
    y_true = test_data_generator.classes
    df = pd.DataFrame(filenames, columns=['filenames'])
    confianca = []
    nb_samples = len(filenames)
    y_preds = model.predict(test_data_generator, steps=nb_samples // batch_size + 1)

    for i in range(len(y_preds)):
        confi = np.max(y_preds[i])
        confianca.append(confi)
        if verbose == 1:
            print('i', i, ' ', y_preds[i])
            print('confidence', confi)

    y_pred = np.argmax(y_preds, axis=1)
    if verbose == 2:
        print('Size y_true', len(y_true))
        print('Size y_pred', len(y_pred))

    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['confidence'] = confianca
    df.insert(loc=2, column='labels', value='')
    df.insert(loc=4, column='predict', value='')
    df.insert(loc=6, column='sit', value='')

    # Check correctness
    for idx, row in df.iterrows():
        cat_true = CATEGORIES[row['y_true']]
        cat_pred = CATEGORIES[row['y_pred']]
        if verbose == 1:
            print('cat_true', cat_true, 'cat_pred', cat_pred)
        df.at[idx, 'labels'] = cat_true
        df.at[idx, 'predict'] = cat_pred
        df.at[idx, 'sit'] = 'C' if row['y_true'] == row['y_pred'] else 'E'

    df = df.sort_values(by='labels')
    df_Err = df[df['sit'] == 'E']
    df_cor = df[df['sit'] == 'C']

    if save_dir:
        df_Err.to_csv(save_dir + 'filterWrong_' + nm_model + '_tempo_' + str(tempo) + '.csv', index=True)
        df_cor.to_csv(save_dir + 'filterCorrect_' + nm_model + '_tempo_' + str(tempo) + '.csv', index=True)

    return y_true, y_pred, df_cor

def predict_unlabeled_data_gen(conf, test_data_generator, model, CATEGORIES, verbose=2):
    """
    Generates predictions for unlabeled data and evaluation metrics (accuracy, precision, recall, fscore, kappa).

    Parameters:
        conf: Configuration dictionary.
        test_data_generator: Images from Image Data Generator.
        model: Loaded model.
        CATEGORIES: List of image classes.
        verbose: Enable printing (0, 1, or 2).

    Returns:
        df: DataFrame with predictions and confidence.
    """
    batch_size = conf['batch_size']

    filenames = test_data_generator.filenames
    y_true = test_data_generator.classes
    df = pd.DataFrame(filenames, columns=['file'])
    confianca = []
    nb_samples = len(filenames)
    y_preds = model.predict(test_data_generator, steps=nb_samples // batch_size + 1)

    for i in range(len(y_preds)):
        confi = np.max(y_preds[i])
        confianca.append(confi)
        if verbose == 1:
            print('i', i, ' ', y_preds[i])
            print('confidence', confi)

    pred = np.argmax(y_preds, axis=1)
    y_pred = [CATEGORIES[y] for y in pred]  # Convert predictions to class names

    df['labels'] = y_pred
    df['confidence'] = confianca

    return df

def metrics(y_true, y_pred):
    """
    Calculates evaluation metrics (accuracy, precision, recall, fscore, kappa).

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        me: Dictionary of metrics.
    """
    # Metrics
    print('\n3-Metrics')
    print('Accuracy, precision, recall, fscore, kappa')
    
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    prec = 3
    me = {
        'precision': round(np.mean(precision, axis=0), prec),
        'recall': round(np.mean(recall, axis=0), prec),
        'fscore': round(np.mean(fscore, axis=0), prec),
        'kappa': round(kappa, prec)
    }

    return me

def class_reports(y_true, y_pred, CATEGORIES, nm_model, save_dir='',
                  tempo=0, verbose=0):
    """
    Generates and optionally saves a classification report.

    Parameters:
        y_true: True labels.
        y_pred: Predicted labels.
        CATEGORIES: List of image classes.
        nm_model: Name of the model.
        save_dir: Path to save the classification report; default is '' (do not save).
        tempo: Time or fold variable for naming.
        verbose: If 1, print the classification report; if 0, do not print.

    Returns:
        None
    """
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=CATEGORIES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    if save_dir:
        df_report.to_csv(save_dir + 'class_report_test_' + nm_model + '_tempo_' + str(tempo) + '.csv', index=True)

def boxplot(nm_model, df_corre, save_dir='', tempo=0, verbose=0):
    """
    Plots and optionally saves a boxplot of confidence scores for correct predictions.

    Parameters:
        nm_model: Name of the model.
        df_corre: DataFrame containing correct predictions.
        save_dir: Path to save the boxplot; default is '' (do not save).
        tempo: Time or fold variable for naming.
        verbose: If 1, print the boxplot; if 0, do not print.

    Returns:
        None
    """
    my_dpi = 100
    plt.figure(figsize=(900/my_dpi, 900/my_dpi), dpi=my_dpi)

    sns.set_style("whitegrid")

    # Adding title to the plot
    sns.boxplot(y=df_corre["labels"], x=df_corre["confidence"])
    plt.title("Classes without Classification Errors", loc="center", fontsize=18)
    plt.xlabel("Accuracy")
    plt.ylabel("Classes")

    if save_dir:
        plt.savefig(save_dir + '/boxplot_correct_' + nm_model + '_tempo_' + str(tempo) + '.jpg')
    if verbose == 1:
        plt.show()
