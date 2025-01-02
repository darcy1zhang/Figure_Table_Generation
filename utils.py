import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


import numpy as np

def get_max_std(me):
    """
    Returns the maximum permissible standard deviation (STD) for a given mean error (ME) 
    based on the table.

    Parameters:
        me (float): The mean error in mmHg.

    Returns:
        float: Maximum permissible STD or None if the value is out of range.
    """
    # Define table rows (mean ranges) and columns (mean error ranges)
    row_values = np.array([0, 1, 2, 3, 4, 5])  # Corresponding to ±0, ±1, ±2, etc.
    column_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    
    # Define the table values (row x column structure)
    table_values = np.array([
        [6.95, 6.95, 6.95, 6.93, 6.92, 6.91, 6.90, 6.90, 6.89, 6.88],
        [6.87, 6.86, 6.84, 6.82, 6.80, 6.78, 6.76, 6.73, 6.71, 6.68],
        [6.65, 6.62, 6.58, 6.55, 6.51, 6.47, 6.43, 6.39, 6.34, 6.30],
        [6.25, 6.20, 6.14, 6.09, 6.03, 5.97, 5.89, 5.83, 5.77, 5.70],
        [5.64, 5.56, 5.49, 5.41, 5.33, 5.25, 5.16, 5.08, 5.01, 4.90],
        [4.79, None, None, None, None, None, None, None, None, None],
    ])
    
    # Convert ME into absolute value and separate into integer and decimal parts
    me_abs = abs(me).astype(np.float32)
    me_abs = round(me_abs, 1)
    me_row = int(me_abs)  # Row is based on the integer part
    me_col = round(me_abs % 1, 1)  # Column is based on the decimal part

    # Check if the ME is out of bounds
    if me_row > 5 or me_col > 0.9:
        return None  # Out of table range
    # Find row and column indices
    # print(me_row, me_col)   
    # print(type(column_values[7]), type(me_col))
    row_index = np.where(row_values == me_row)[0][0]
    col_index = np.where(column_values == me_col)[0][0]
    
    # Retrieve the value from the table
    max_std = table_values[row_index, col_index]
    
    return max_std

def bland_altman_plot(data, name):
    """
    Generate a Bland-Altman plot to visualize the difference between predictions and labels.

    Args:
        data: data set containing predictions and labels
        name: name of the vital signal (e.g., SP_on_sample_level)

    Outputs:
        Saves the Bland-Altman plot as a PNG file in the './Fig/' directory.
    """
    # Calculate the mean and difference between predictions and labels
    pred = data[:, 1].flatten()
    label = data[:, 2].flatten()

    mean = np.mean([pred, label], axis=0)
    diff = pred - label
    mean_diff = np.mean(diff)  # Mean of the differences
    std_diff = np.std(diff)    # Standard deviation of the differences

    # Calculate density for scatter plot
    xy = np.vstack([mean, diff])       # Stack mean and diff for density calculation
    density = gaussian_kde(xy)(xy)     # Kernel density estimation
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3B3EB", "#3636FF", "#00006C"])

    # Create the Bland-Altman plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mean, diff, c=density, cmap=custom_cmap, s=10)  # Scatter plot with density
    plt.colorbar(scatter, label='Density')  # Add color bar for density
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Difference ({mean_diff:.2f})')  # Mean line
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='-.', label=f'+1.96 SD ({mean_diff + 1.96 * std_diff:.2f})')  # +1.96 SD line
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label=f'-1.96 SD ({mean_diff - 1.96 * std_diff:.2f})')  # -1.96 SD line
    
    # Add labels, title, and legend
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements')
    name_with_space = name.replace('_', ' ')
    plt.title(f'Bland-Altman Plot of {name_with_space}')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('./Fig/Bland_Altman_Plot_' + name + '.png')

def trend_plot(data, name):
    """
    Generate a trend plot to visualize how predictions and labels match each other.

    Args:
        data: data set containing predictions and labels

    Outputs:
        Saves the trend plot as a PNG file in the './Fig/' directory.
    """

    pred = data[:, 1].flatten()
    label = data[:, 2].flatten()

    # Sort predictions and labels based on the order of labels
    index = np.argsort(label)
    pred = pred[index]
    label = label[index]

    # Calculate metrics
    mae = np.mean(np.abs(pred - label))  # Mean Absolute Error
    me = np.mean(pred - label)          # Mean Error
    correlation_matrix = np.corrcoef(pred, label)  # Correlation coefficient matrix
    correlation = correlation_matrix[0, 1]        # Extract correlation value
    std = np.std(pred - label)          # Standard deviation of differences

    # Calculate density for scatter plot
    xy = np.vstack([label, pred])       # Stack labels and predictions for density calculation
    density = gaussian_kde(xy)(xy)     # Kernel density estimation
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#B3B3EB", "#3636FF", "#00006C"])

    # Create the trend plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(np.arange(1, pred.shape[0]+1), pred, c=density, cmap=custom_cmap, s=1)  # Scatter plot of predictions
    plt.scatter(np.arange(1, pred.shape[0]+1), label, c="red", s=1)                            # Scatter plot of labels
    plt.colorbar(scatter, label='Density')  # Add color bar for density
    plt.legend(["Prediction", "Label"], loc="upper left")
    
    # Add text with metrics in the plot
    plt.text(1, 0, f"MAE:{mae:6.2f} ME:  {me:6.2f}\nSTD: {std:6.2f} Corr: {correlation:3.2f}", 
             fontsize="x-large", ha='right', va='bottom', transform=plt.gca().transAxes)
    
    # Add title and save the plot
    name_with_space = name.replace('_', ' ')
    plt.title(f'Trend Plot of {name_with_space}')
    plt.savefig('./Fig/Trend_Plot_' + name + '.png')

def generate_pdf(vital_signals_):
    """
    Calculate metrics from prediction and label arrays, and display as a table.

    Args:
        vital_signal (str): Vital signal name (e.g., 'SP', 'DP', 'HR', 'RR')
    """

    vital_signals = [x.strip() for x in vital_signals_.split(',')]
    print(vital_signals)
    dict_list = []
    dict_ID_list = []

    for vital_signal in vital_signals:
        if os.path.exists(f'./data/{vital_signal}_results.npy'):
            data = np.load(f'./data/{vital_signal}_results.npy')
        elif os.path.exists(f'./data/{vital_signal}_results.csv'):
            data = pd.read_csv(f'./data/{vital_signal}_results.csv')
            data_tmp1 = data['ID'].to_numpy().reshape(-1, 1)
            data_tmp2 = data['pred'].to_numpy().reshape(-1, 1)
            data_tmp3 = data['label'].to_numpy().reshape(-1, 1)
            data = np.hstack((data_tmp1, data_tmp2, data_tmp3))
        else:
            raise ValueError(f"Can't find the data file: {vital_signal}_results.npy/csv")

        data_ID_subject = np.unique(data[:, 0])
        data_pred_subject = []
        data_label_subject = []

        for i in np.unique(data_ID_subject):
            data_subject_tmp = data[data[:, 0] == i]
            data_pred_subject.append(np.mean(data_subject_tmp[:, 1]))
            data_label_subject.append(np.mean(data_subject_tmp[:, 2]))
        
        data_ID_subject = data_ID_subject.reshape(-1, 1)
        data_pred_subject = np.array(data_pred_subject).reshape(-1, 1)
        data_label_subject = np.array(data_label_subject).reshape(-1, 1)
        data_subject = np.hstack((data_ID_subject, data_pred_subject, data_label_subject))

        trend_plot(data, vital_signal + '_on_sample_level')
        bland_altman_plot(data, vital_signal + '_on_sample_level')
        bland_altman_plot(data_subject, vital_signal + '_on_subject_level')

        # Compute metrics
        sample_level_mae = np.mean(np.abs(data[:, 1] - data[:, 2]))  # Mean Absolute Error
        sample_level_me = np.mean(data[:, 1] - data[:, 2])          # Mean Error
        sample_level_correlation = np.corrcoef(data[:, 1], data[:, 2])[0, 1]  # Correlation coefficient
        sample_level_std = np.std(data[:, 1] - data[:, 2])          # Standard deviation of errors

        subject_level_mae = np.mean(np.abs(data_subject[:, 1] - data_subject[:, 2]))  # Mean Absolute Error
        subject_level_me = np.mean(data_subject[:, 1] - data_subject[:, 2])          # Mean Error
        subject_level_correlation = np.corrcoef(data_subject[:, 1], data_subject[:, 2])[0, 1]  # Correlation coefficient
        subject_level_std = np.std(data_subject[:, 1] - data_subject[:, 2])          # Standard deviation of errors

        dict_tmp = {}
        dict_tmp['vital_signal'] = vital_signal + '_on_sample_level'
        dict_tmp['mae'] = sample_level_mae
        dict_tmp['me'] = sample_level_me
        dict_tmp['correlation'] = sample_level_correlation
        dict_tmp['std'] = sample_level_std
        dict_list.append(dict_tmp)
        dict_tmp = {}
        dict_tmp['vital_signal'] = vital_signal + '_on_subject_level'
        dict_tmp['mae'] = subject_level_mae
        dict_tmp['me'] = subject_level_me
        dict_tmp['correlation'] = subject_level_correlation
        dict_tmp['std'] = subject_level_std
        dict_list.append(dict_tmp)

        tex_file = rf"""
\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{float}}

\begin{{document}}
"""
    for vital_signal in vital_signals:
        vital_signal_name = vital_signal.replace('_', ' ')
        tex_file += rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{./Fig/Trend_Plot_{vital_signal}_on_sample_level.png}}
\caption{{Trend Plot of {vital_signal_name} on sample level.}}
\label{{fig:image1}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{./Fig/Bland_Altman_Plot_{vital_signal}_on_sample_level.png}}
\caption{{Bland-Altman plot of all subjects' {vital_signal_name} prediction vs label measurements. Here one dot represents one measurement pair.}}
\label{{fig:image1}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{./Fig/Bland_Altman_Plot_{vital_signal}_on_subject_level.png}}
\caption{{Bland-Altman plot of each subject's averaged {vital_signal_name} prediction vs label measurements. Here one dot represents one subject.}}
\label{{fig:image1}}
\end{{figure}}
"""
    tex_file += rf"""

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{./Fig/ME_STD_Correspond.png}}
\caption{{Averaged subject data acceptance in mmHg}}
\label{{ME_STD}}
\end{{figure}}

\begin{{table}}[h!]
\centering
\begin{{tabular}}{{|c|c|c|c|c|c|}}
\hline
Vital Signal & ME & MAE & SD & Correlation & Whether meet the requirement\\ \hline

"""
    for ind in range(len(dict_list)):
        flag = None
        if dict_list[ind]['vital_signal'].find('subject') == -1:
            # sample level
            if (dict_list[ind]['me'] < 5) and (dict_list[ind]['std'] < 8):
                flag = "Yes"
            else:
                flag = "No"    
        else:
            # subject level
            required_std = get_max_std(dict_list[ind]['me'])
            print(dict_list[ind]['me'], required_std, dict_list[ind]['std'])
            if (required_std > dict_list[ind]['std']) and dict_list[ind]['me'] < 5:
                flag = "Yes"
            else:
                flag = "No"
        tex_file += rf"""
{dict_list[ind]['vital_signal'].replace('_', ' ')} & {dict_list[ind]['me']:.2f} & {dict_list[ind]['mae']:.2f} & {dict_list[ind]['std']:.2f} & {dict_list[ind]['correlation']:.2f} & {flag} \\ \hline
"""
    tex_file += rf"""

\end{{tabular}}
\caption{{Prediction Results}}
\label{{tab:metrics}}
\end{{table}}

\end{{document}}
"""

    # Print the LaTeX table
    # print(latex_table)
    with open('./Tex/results.tex', 'w') as tex:
        tex.write(tex_file)

    os.system("pdflatex ./Tex/results.tex")
    os.system("rm ./results.aux ./results.log")
    os.system("mv ./results.pdf ./pdf/")