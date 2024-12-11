import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Figures as FDA Requirement')
    parser.add_argument('--vital_signal', type=str, default='SP, DP, HR, RR',
                        help='Vital signals which you want to plot, like SP, DP, HR, RR') 
    args = parser.parse_args()
    generate_pdf(args.vital_signal)