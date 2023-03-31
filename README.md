# Symbolic Regression GECCO Competition - 2023 - Track 1 - Performance

The participants will be free to experiment with these data sets until the deadline. 
Analysis on each dataset should include the production of a single best model for each dataset, and an extended abstract discussing the pipeline.

At the competition submission deadline the repository of a participating team should contain:

- [**required**] A file containing a single model as a **sympy-compatible expression**, selected as the best expression for that data set, named `dataset_X_best_model`.
- [**required**] A maximum 4 page extended abstract in PDF format describing the algorithm and pipeline. This PDF must contain the name and contact information of the participants.
- [to be eligible for prize] Reproducibility documentation in the `src` folder.
    - Installation and execution instructions 
    - Scripts (Jupyter Notebook is accepted) with detailed instructions of all steps used to produce models (including hyperparameters search, if any) 

## Evaluation criteria

The final score of each competitor will be composed of:

- *acc*: Rank based on the accuracy (R^2) on a separate validation set for each data set.
- *simpl*: Rank based on the a simplicity (number of nodes calculating by traversing the sympy expression) of the model for each data set.

The rank will be calculated for each data set independently such that, with N participants, the k-th ranked competitor (k=1 being the best) will be assigned a value of *N - k + 1*. The final score will be the harmonic mean of all of the scores and each participant will be ranked accordingly:

```python
score = 2*n / sum([ (1/acc[i]) + (1/simpl[i]) for i in (1..n)])
```

## Suporting scripts

Your repository will contain a script called `scores_test.py` that can be used to calculate the training set scores and verify that everything works as intended. This script will be used to determine the scores of each team on the test set and, subsequently, the ranks and final score. We also provide three files containing dummy models for each dataset to serve as an example. Please replace those with the real models.

## Repository License

The repositories will be kept private during the whole competition and it will become open to the public **after** GECCO 2023 conference with a BSD-3 license. Please, make sure that you only keep files conforming to such license.

## Deadline

01 June 2023, 00:00 anywhere in the world.

## Question and issues

Any questions and issues can be addressed to folivetti@ufabc.edu.br or at our Discord server (https://discord.gg/Dahqh3Chwy)