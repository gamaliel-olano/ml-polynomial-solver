# tinygrad-sgd-polynomial-solver

This polynomial uses SGD in the tinygrad framework to predict the degree and coefficients of a dataset. A model is trained for each possible value of the degree (<5) and the one with the lowest loss becomes the predicted degree.

Run by using the command: python3 solver.py

Argparse was used with default values for the following variables: epoch, lr, size

Make sure that data_train.csv and data_test.csv files are on the same folder as solver.py

# Update

Added example.ipynb which shows how the code runs without using the argparse function.

solver.py was also modified to fix the perceived errors

# Assignment Instructions
Objective 

SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients.

Usage:

The application will be used as follows:

python3 solver.py

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv.

The function should be modeled using tinygrad : https://github.com/geohot/tinygrad

Use SGD to learned the polynomial coefficients.
