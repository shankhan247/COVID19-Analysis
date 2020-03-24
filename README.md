# Exploring Coronavirus data 

This repository contains some basic set up for exploration of the JHU data regarding
the Coronavirus pandemic of 2020.

## Setting up

Create a conda environment and activate it:

```
conda env create -f env.yml
conda activate coronavirus
```

## Get the data

```
git submodule update --init --recursive
```

The data is now at `COVID-19`. The data is updated daily so you can always keep pulling
new data in as it becomes available.

## How To Use Code

Code is located in exp folder. Code is described below. You can run code in terminal. Linux command is "python insert_filename_here". 

- `exp/regr.py`: Trains a linear regression model on the confirmed Corona cases within the  US and predicts future case numbers for the next 11 days. Generates 3 figures: confirmed cases vs predicted cases, trajectory of known and predicted cases over each day since Jan 22 2020, and rates of case numbers per day.
- `exp/regr_italy.py`: Same as regr.py except a polynomial regression model is used and the cases in Italy are examined. 
- `exp/regr_china.py`: Same as regr_italy.py except the cases in China are examined.
