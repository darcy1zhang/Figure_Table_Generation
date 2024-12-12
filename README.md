# Figures and Tables Generation

Note: Only Bland-Altman Figure is required; the Trend Figure is for reference only.

To convert LaTeX source files (.tex) directly into PDF files, you should hava the command-line tool called "pdflatex" first. For linux, you can type `sudo apt install texlive`; For windows, you can visit website of TeX Live to download it.

Example:
```
pip install -r requirements.txt # install the necessary packages
python plot.py --vital_signal 'SP, DP, HR, RR'
```

All the result file should be stored in the folder "./data/". And the name of the file should be "SP/DP/HR/RR_results.npy/csv".