import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Figures as FDA Requirement')
    parser.add_argument('--sample_level_path', type=str, default=None,
                        help='Path for predictions and labels of sample level, like ./data/DP_sample_level.npy')
    parser.add_argument('--subject_level_path', type=str, default=None,
                        help='Path for predictions and labesl of subject level, like ./data/DP_subject_level.csv')   
    parser.add_argument('--name', type=str, default='D_on_sample_level',
                        help='S/D/HR/RR on sample-level/subject-level')
    args = parser.parse_args()


    bland_altman_plot(args.sample_level_path)
    bland_altman_plot(args.subject_level_path)
    trend_plot(args.sample_level_path)
    generate_pdf(args.sample_level_path, args.subject_level_path, args.name)