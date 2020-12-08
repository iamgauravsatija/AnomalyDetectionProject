# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Anomaly detection algorithm using guassian dirtibution formula
# for a 2 feature dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class guassian_algorithm:
    def __init__(self, file):
        self.file = file

    def calc_probability(file):
        total_records = len(file)  # total number of rows or data points
        sum = np.sum(file, axis=0)  # sum of all the values n first column and second column separately
        mean = sum/total_records  # mean of both the columns

        variance_denominator = np.sum((file - mean)*2, axis=0)
        variance_square = variance_denominator/mean  # It is important to note it is variance square

        variance_diag = np.diag(variance_square)  # diagonal values of the variance
        variance_sq_determinant = np.linalg.det(variance_diag) # compute the determinant

        mean_len = len(mean)
        X = file - mean
        prob = 1/((2*np.pi)*(variance_sq_determinant**0.5))*np.exp(-0.5*np.sum(X @ np.linalg.pinv(variance_diag)*X, axis=1))
        # np.linalg.pinv computes sudo inverse of matrix
        # credit for above line:
        # https://towardsdatascience.com/a-complete-anomaly-detection-algorithm-from-scratch-in-python-step-by-step-guide-e1daf870336e

        return prob

    def algorithm(self):
        # total_records = len(self.file)  # total number of rows or data points
        # sum = np.sum(file, axis=0)  # sum of all the values n first column and second column separately
        # mean = sum/total_records  # mean of both the columns
        #
        # variance_denominator = np.sum((file - mean)**2, axis=0)
        # variance_square = variance_denominator/mean  # It is important to note it is variance square
        #
        # variance_diag = np.diag(variance_square)  # diagonal values of the variance
        # variance_sq_determinant = np.linalg.det(variance_diag) # compute the determinant
        #
        # mean_len = len(mean)
        # X = file - mean
        # prob = 1/((2*np.pi)*(variance_sq_determinant**0.5))*np.exp(-0.5*np.sum(X @ np.linalg.pinv(variance_diag)*X, axis=1))
        # # np.linalg.pinv computes sudo inverse of matrix
        # # credit for above line:
        # # https://towardsdatascience.com/a-complete-anomaly-detection-algorithm-from-scratch-in-python-step-by-step-guide-e1daf870336e


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Excel file used in this example has 3 sheets (X, Xval & y)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Read the file using panda library function

        # Read the main sheet i.e. X
        file = pd.read_excel('/Users/gauravsatija/Desktop/CPSC473-Data-Mining/Data_Mining_Project/ex8data1.xlsx', sheet_name='X', header=None)
        prob_x = self.calc_probability(file)

        # Read the cross-validation data
        file = pd.read_excel('/Users/gauravsatija/Desktop/CPSC473-Data-Mining/Data_Mining_Project/ex8data1.xlsx', sheet_name='Xval', header=None)
        prob_xval = self.calc_probability(file)












# First read the excel file which represents a two feature dataset
# The 2 features can be thought of as values on x-axis and y-axis
# In total there are 307 data points with 2 features in the excel file.

# Read the file using panda library function
file = pd.read_excel('/Users/gauravsatija/Desktop/CPSC473-Data-Mining/Data_Mining_Project/ex8data1.xlsx', sheet_name='X', header=None)

# print(file) # prints the dataset

# Plot the dataset on a 2D graph using the matplotlib.pyplot library
plt.figure()
plt.scatter(file[0], file[1])  # 1st column as x-axis and 2nd column as y-axis
plt.show()
