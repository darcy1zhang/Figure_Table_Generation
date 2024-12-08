import argparse
from utils import *

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
    generate_metrics_table(args.pred_path, args.label_path)