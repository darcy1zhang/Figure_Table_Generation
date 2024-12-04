import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde

def bland_altman_plot(pred, label, name):
    mean = np.mean([pred, label], axis=0)
    diff = pred - label
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(mean_diff, color='gray', linestyle='--', label=f'Mean Difference ({mean_diff:.2f})') # mean_diff:.2f
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label=f'+1.96 SD ({mean_diff + 1.96 * std_diff:.2f})')
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', label=f'-1.96 SD ({mean_diff - 1.96 * std_diff:.2f})')
    
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements')
    plt.title(f'Bland-Altman Plot of {name}')
    plt.legend()
    plt.savefig('./Fig/Bland_Altman_Plot_' + name + '.png')

def trend_plot(pred, label, name):

    index = np.argsort(label)
    pred = pred[index]
    label = label[index]
    mae = np.mean(np.abs(pred - label))
    me = np.mean(pred - label)
    correlation_matrix = np.corrcoef(pred, label)
    correlation = correlation_matrix[0, 1]
    std = np.std(pred - label)
    xy = np.vstack([label, pred])
    density = gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(np.arange(1, pred.shape[0]+1), pred, c=density, cmap='viridis', s=1)
    plt.scatter(np.arange(1, pred.shape[0]+1), label, c="blue", s=1)
    plt.colorbar(scatter, label='Density')
    plt.legend(["Prediction", "Label"], loc="upper left")
    plt.text(1, 0, f"MAE:{mae:6.2f} ME:  {me:6.2f}\nSTD: {std:6.2f} Corr: {correlation:3.2f}", fontsize="x-large", ha='right', va='bottom', transform=plt.gca().transAxes)
    plt.title(f'Trend Plot of {name}')
    plt.savefig('./Fig/Trend_Plot_' + name + '.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Figures as FDA Requirement')
    parser.add_argument('--label_path', type=str, default='./data/label.npy',
                        help='Path for labels')
    parser.add_argument('--pred_path', type=str, default='./data/pred.npy',
                        help='Path for predictions')   
    parser.add_argument('--name', type=str, default='D on sample-level',
                        help='S/D/HR/RR on sample-level/subject-level')
    args = parser.parse_args()

    label = np.load(args.label_path)
    pred = np.load(args.pred_path)

    bland_altman_plot(pred, label, args.name)
    trend_plot(pred, label, args.name)