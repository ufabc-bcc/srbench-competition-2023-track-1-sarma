# RILS -- Regression via Iterated Local Search
RILS algorithm for GECCO2023 SR competition

# Installation instructions

0. The program requires Python 3 (tested on version 3.11.3 under Windows 10), but it should work on some earlier versions as well. 
We also recommend using pip package manager. 

1. Download the repository and unpack it. 

2. Install the following pip packages:

```console
pip install numpy
pip install sympy
pip install scikit-learn
pip install statsmodels
```

# Execution instructions

1. Position inside the src/ directory. The directories at this level are: instances/ results/ and rils/.
Directory instances/ holds datasets that are previously rounded to 9 decimals and saved in the tab separated .txt files.  
Directory results/ holds execution results obtained on Windows 10 OS -- Linux OS was not tested. 
Directory rils/ holds program codes. 

2. Call run_all.cmd on Windows OS. 
The content of run_all.cmd file is as follows:

```console
python run.py "instances/srbench_2023" "dataset_1.txt" 180 20
python run.py "instances/srbench_2023" "dataset_2.txt" 100 10
python run.py "instances/srbench_2023" "dataset_3.txt" 180 20
```

3. The steps in the execution pipeline are explained in ../paper/rils_gecco2023.pdf. 
Briefly, several output files will occurr during execution:

    1. best_sols_dataset_{1|2|3}.txt
    2. log.txt
    3. out.txt

The model will be written inside out.txt.
Note that there is semi-automated (interactive) simplification (step 4 in the paper) for the dataset_3. 
