# SIMLR-python
***[1] Wang, Bo, et al. "Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning." Nature methods 14.4 (2017): 414.***
## Using Instruction
Before running demo.py, please ensure that your Python is version 3.7.1, and NumPy v1.15.4, scipy v1.1.0, scikit-learn v0.20.1, or these codes may not work well.

If you first run these codes, please enter src directory and run setup.py using command
`python setup.py build`
Then a sub-directory named 'build' will be automatically produced. Copy all the files in '/build/lib...' into the directory src. *This procedure is to help compile all the C/C++ files.*

All the data sets are saved in .pkl files which is convenient to use in python. Their original data are saved in .csv files.

## Overview of the SIMLR algorithm
![](/SIMLR-pic/SIMLR-pic.001 "001")
![](/SIMLR-pic/SIMLR-pic.002 "002")
![](/SIMLR-pic/SIMLR-pic.003 "003")
![](/SIMLR-pic/SIMLR-pic.004 "004")
![](/SIMLR-pic/SIMLR-pic.005 "005")