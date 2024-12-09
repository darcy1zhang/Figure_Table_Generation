import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

def bland_altman_plot(pred, label, name):
    """
    Generate a Bland-Altman plot to visualize the difference between predictions and labels.

    Args:
        pred (array-like): Predicted values.
        label (array-like): Ground truth values (labels).
        name (str): Name to use in the plot title and output file.

    Outputs:
        Saves the Bland-Altman plot as a PNG file in the './Fig/' directory.
    """
    # Calculate the mean and difference between predictions and labels
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

def trend_plot(pred, label, name):
    """
    Generate a trend plot to visualize how predictions and labels match each other.

    Args:
        pred (array-like): Predicted values.
        label (array-like): Ground truth values (labels).
        name (str): Name to use in the plot title and output file.

    Outputs:
        Saves the trend plot as a PNG file in the './Fig/' directory.
    """
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

def generate_pdf(pred_path, label_path, name, metrics=['ME', 'SD', 'SD', 'Correlation'], output_latex=True):
    """
    Calculate metrics from prediction and label arrays, and display as a table.

    Args:
        pred_path (str): Path to the .npy file for predictions.
        label_path (str): Path to the .npy file for labels.
        metrics (list): List of metrics to compute. It can be one of ['ME', 'SD', 'MAE', 'Correlation']
        output_latex (bool): If True, output the LaTeX format of the table.
    """
    # Load the prediction and label arrays
    pred = np.load(pred_path)
    label = np.load(label_path)

    # Ensure the arrays are of the same length
    if pred.shape != label.shape:
        raise ValueError("The shapes of pred and label arrays must match!")

    # Compute metrics
    mae = np.mean(np.abs(pred - label))  # Mean Absolute Error
    me = np.mean(pred - label)          # Mean Error
    correlation = np.corrcoef(pred, label)[0, 1]  # Correlation coefficient
    std = np.std(pred - label)          # Standard deviation of errors

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    if 'MAE' in metrics:
        table.add_row(["MAE", f"{mae:.2f}"])
    if 'ME' in metrics:
        table.add_row(["ME", f"{me:.2f}"])
    if 'Correlation' in metrics:
        table.add_row(["Correlation", f"{correlation:.2f}"])
    if 'SD' in metrics:        
        table.add_row(["SD", f"{std:.2f}"])

    # Display the table
    print("Metrics Table:")
    print(table)

    # Generate LaTeX table if requested
    if output_latex:
        # Create a dictionary of metrics and their values
        metrics_dict = {
            "MAE": mae,
            "ME": me,
            "Correlation": correlation,
            "SD": std
        }

        # Build the LaTeX table dynamically
        latex_table = rf"""
\documentclass{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage{{graphicx}}
\usepackage{{float}}

\begin{{document}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{./Fig/Trend_Plot_{name}.png}}
\caption{{Trend Plot}}
\label{{fig:image1}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{./Fig/Bland_Altman_Plot_{name}.png}}
\caption{{Bland Altman Plot}}
\label{{fig:image2}}
\end{{figure}}

\begin{{table}}[h!]
\centering
\begin{{tabular}}{{|c|c|}}
\hline
Metric & Value \\ \hline
"""

        for metric in metrics:
            if metric in metrics_dict:
                latex_table += f"{metric} & {metrics_dict[metric]:.2f} \\\\ \\hline\n"

        latex_table += r"""
\end{tabular}
\caption{Prediction Results}
\label{tab:metrics}
\end{table}

\end{document}
"""

    # Print the LaTeX table
    print(latex_table)
    with open('./Tex/results.tex', 'w') as tex_file:
        tex_file.write(latex_table)

    os.system("pdflatex ./Tex/results.tex")