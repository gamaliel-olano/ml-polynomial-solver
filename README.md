# tinygrad-sgd-polynomial-solver

This polynomial uses SGD in the tinygrad framework to predict the degree and coefficients of a dataset. A model is trained for each possible value of the degree (<5) and the one with the lowest loss becomes the predicted degree.

Run by using the command: python3 solver.py

Argparse was used with default values for the following variables: epoch, lr, size

Make sure that data_train.csv and data_test.csv files are on the same folder as solver.py

# Update

Added example.ipynb which shows how the code runs without using the argparse function.

solver.py was also modified to fix the perceived errors
