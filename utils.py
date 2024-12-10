import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import pandas as pd

def bland_altman_plot(path):
    """
    Generate a Bland-Altman plot to visualize the difference between predictions and labels.

    Args:
        path: where the data stored

    Outputs:
        Saves the Bland-Altman plot as a PNG file in the './Fig/' directory.
    """
    # Calculate the mean and difference between predictions and labels
    if path.endswith('.npy'):
        data = np.load(path)
        pred = data[:, 0].flatten()
        label = data[:, 1].flatten()
    elif path.endswith('.csv'):
        data = pd.read_csv(path) 
        pred = data['pred'].to_numpy().flatten()
        label = data['label'].to_numpy().flatten()
    else:
        raise ValueError("Data format should be csv or npy")
    tmp = path.split('/')[-1]
    part1 = tmp.split('_')[0]  # 'SP, DP, HR, RR'
    part2 = tmp.split('_')[1]  # 'sample/subject'
    name = str(part1) + '_' + str(part2)
    # print(name)
    
    mean = np.mean([pred, label], axis=0)
    diff = pred - label
    mean_diff = np.mean(diff)  # Mean of the differences
    std_diff = np.std(diff)    # Standard deviation of the differences
    
    # Create the Bland-Altman plot
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)  # Scatter plot of mean vs. difference
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Difference ({mean_diff:.2f})')  # Mean line
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label=f'+1.96 SD ({mean_diff + 1.96 * std_diff:.2f})')  # +1.96 SD line
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', label=f'-1.96 SD ({mean_diff - 1.96 * std_diff:.2f})')  # -1.96 SD line
    
    # Add labels, title, and legend
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements')
    plt.title(f'Bland-Altman Plot of {name}')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('./Fig/Bland_Altman_Plot_' + name + '.png')

def trend_plot(path):
    """
    Generate a trend plot to visualize how predictions and labels match each other.

    Args:
        path: where the data stored

    Outputs:
        Saves the trend plot as a PNG file in the './Fig/' directory.
    """

    if path.endswith('.npy'):
        data = np.load(path)
        pred = data[:, 0].flatten()
        label = data[:, 1].flatten()
    elif path.endswith('.csv'):
        data = pd.read_csv(path) 
        pred = data['pred'].to_numpy().flatten()
        label = data['label'].to_numpy().flatten()
    else:
        raise ValueError("Data format should be csv or npy")
    
    tmp = path.split('/')[-1]
    part1 = tmp.split('_')[0]  # 'SP, DP, HR, RR'
    part2 = tmp.split('_')[1]  # 'sample/subject'
    name = str(part1) + '_' + str(part2)

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

    # Create the trend plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(np.arange(1, pred.shape[0]+1), pred, c=density, cmap='viridis', s=1)  # Scatter plot of predictions
    plt.scatter(np.arange(1, pred.shape[0]+1), label, c="blue", s=1)                            # Scatter plot of labels
    plt.colorbar(scatter, label='Density')  # Add color bar for density
    plt.legend(["Prediction", "Label"], loc="upper left")
    
    # Add text with metrics in the plot
    plt.text(1, 0, f"MAE:{mae:6.2f} ME:  {me:6.2f}\nSTD: {std:6.2f} Corr: {correlation:3.2f}", 
             fontsize="x-large", ha='right', va='bottom', transform=plt.gca().transAxes)
    
    # Add title and save the plot
    plt.title(f'Trend Plot of {name}')
    plt.savefig('./Fig/Trend_Plot_' + name + '.png')

def generate_pdf(sample_level_path, subject_level_path, name, metrics=['ME', 'MAE', 'SD', 'Correlation'], output_latex=True):
    """
    Calculate metrics from prediction and label arrays, and display as a table.

    Args:
        pred_path (str): Path to the .npy file for predictions.
        label_path (str): Path to the .npy file for labels.
        metrics (list): List of metrics to compute. It can be one of ['ME', 'SD', 'MAE', 'Correlation']
        output_latex (bool): If True, output the LaTeX format of the table.
    """

    tmp = sample_level_path.split('/')[-1]
    part1 = tmp.split('_')[0]  # 'SP, DP, HR, RR'
    part2 = tmp.split('_')[1]  # 'sample/subject'
    name = str(part1) + '_' + str(part2)

    sample_level = 0
    subject_level = 0
    # Load the prediction and label arrays
    if sample_level_path != None:
        sample_level = 1
        if sample_level_path.endswith('.npy'):
            data = np.load(sample_level_path)
            sample_level_pred = data[:, 0].flatten()
            sample_level_label = data[:, 1].flatten()
        elif sample_level_path.endswith('.csv'):
            data = pd.read_csv(sample_level_path)  
            sample_level_pred = data['pred'].to_numpy().flatten()
            sample_level_label = data['label'].to_numpy().flatten()
        else:
            raise ValueError("Data format should be csv or npy")

    if subject_level_path != None:
        subject_level = 1
        if subject_level_path.endswith('.npy'):
            data = np.load(subject_level_path)
            subject_level_pred = data[:, 0].flatten()
            subject_level_label = data[:, 1].flatten()
        elif subject_level_path.endswith('.csv'):
            data = pd.read_csv(subject_level_path)  # 没有列名时用 header=None
            subject_level_pred = data['pred'].to_numpy().flatten()
            subject_level_label = data['label'].to_numpy().flatten()
        else:
            raise ValueError("Data format should be csv or npy")


    # Ensure the arrays are of the same length
    if sample_level and sample_level_pred.shape != sample_level_label.shape:
        raise ValueError("The shapes of pred and label arrays of sample level must match!")
    
    if subject_level and subject_level_pred.shape != subject_level_label.shape:
        raise ValueError("The shapes of pred and label arrays of subject level must match!")

    # Compute metrics
    sample_level_mae = np.mean(np.abs(sample_level_pred - sample_level_label))  # Mean Absolute Error
    sample_level_me = np.mean(sample_level_pred - sample_level_label)          # Mean Error
    sample_level_correlation = np.corrcoef(sample_level_pred, sample_level_label)[0, 1]  # Correlation coefficient
    sample_level_std = np.std(sample_level_pred - sample_level_label)          # Standard deviation of errors

    subject_level_mae = np.mean(np.abs(subject_level_pred - subject_level_label))  # Mean Absolute Error
    subject_level_me = np.mean(subject_level_pred - subject_level_label)          # Mean Error
    subject_level_correlation = np.corrcoef(subject_level_pred, subject_level_label)[0, 1]  # Correlation coefficient
    subject_level_std = np.std(subject_level_pred - subject_level_label)          # Standard deviation of errors

    # Create a PrettyTable
    table = PrettyTable()
    # table.field_names = ["Metric", "Value"]
    # if 'MAE' in metrics:
    #     table.add_row(["MAE", f"{mae:.2f}"])
    # if 'ME' in metrics:
    #     table.add_row(["ME", f"{me:.2f}"])
    # if 'Correlation' in metrics:
    #     table.add_row(["Correlation", f"{correlation:.2f}"])
    # if 'SD' in metrics:        
    #     table.add_row(["SD", f"{std:.2f}"])
    table.field_names = ["Level", "ME", "MAE", "SD", "Correlation"]
    table.add_row(['Sample Level', f"{sample_level_me:.2f}", f"{sample_level_mae:.2f}", f"{sample_level_std:.2f}", f"{sample_level_correlation:.2f}"])
    table.add_row(['Subject Level', f"{sample_level_me:.2f}", f"{sample_level_mae:.2f}", f"{sample_level_std:.2f}", f"{sample_level_correlation:.2f}"])

    # Display the table
    print("Metrics Table:")
    print(table)

    # Generate LaTeX table if requested
    if output_latex:
        # Create a dictionary of metrics and their values
        # metrics_dict = {
        #     "MAE": mae,
        #     "ME": me,
        #     "Correlation": correlation,
        #     "SD": std
        # }

        # Build the LaTeX table dynamically
        latex_table = rf"""
\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{float}}

\begin{{document}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{./Fig/Trend_Plot_{part1}_sample.png}}
\caption{{Trend Plot}}
\label{{fig:image1}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{./Fig/Bland_Altman_Plot_{part1}_subject.png}}
\caption{{Bland Altman Plot of subject level}}
\label{{fig:image2}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{./Fig/Bland_Altman_Plot_{part1}_sample.png}}
\caption{{Bland Altman Plot of sample level}}
\label{{fig:image2}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{./Fig/ME_STD_Correspond.png}}
\caption{{Averaged subject data acceptance in mmHg}}
\label{{ME_STD}}
\end{{figure}}

\begin{{table}}[h!]
\centering
\begin{{tabular}}{{|c|c|c|c|c|}}
\hline
Level & ME & MAE & SD & Correlation \\ \hline
Sample Level & {sample_level_me:.2f} & {sample_level_mae:.2f} & {sample_level_std:.2f} & {sample_level_correlation:.2f} \\ \hline
Subject Level & {subject_level_me:.2f} & {subject_level_mae:.2f} & {subject_level_std:.2f} & {subject_level_correlation:.2f} \\ \hline
\end{{tabular}}
\caption{{Prediction Results}}
\label{{tab:metrics}}
\end{{table}}

\end{{document}}
"""

    # Print the LaTeX table
    print(latex_table)
    with open('./Tex/results.tex', 'w') as tex_file:
        tex_file.write(latex_table)

    os.system("pdflatex ./Tex/results.tex")

# """

#         for metric in metrics:
#             if metric in metrics_dict:
#                 latex_table += f"{metric} & {metrics_dict[metric]:.2f} \\\\ \\hline\n"

#         latex_table += r"""