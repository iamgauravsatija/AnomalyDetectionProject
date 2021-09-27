from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.ScoreCalculator import ScoreCalculator


class GuassianAlgorithm:
    def __init__(self, file_path='data/ex8data1.xlsx') -> None: 
        self.int = 0
        self.file_path = file_path

    def guassian_algorithm(self) -> None:
        """
        - Reads the given data file from sheet X and Xval
        - Calculates the probability of each data point
        - Get the epsilon list of data points
        - Calculates the f-score using epsilon and probability values
        - Based on f-score determines which points are outliers (anomplous)
        - Plot the graph

        Useful links:
        - https://deepai.org/machine-learning-glossary-and-terms/f-score
        """

        # Reading main data point sheet
        file_x = pd.read_excel(self.file_path, sheet_name='X', header=None)
        prob_x = self.__probability(file_x)        
        
        # here we are going to read the cross validation data to calculate the probability
        file_cv = pd.read_excel(self.file_path, sheet_name='Xval', header=None)
        prob_cv = self.__probability(file_cv)


        # Read labels file
        # Labels file is used as the ground truth to check if the training model is working good or not
        file_label = pd.read_excel(self.file_path, sheet_name='y', header=None)
        array_label = np.array(file_label)  # converting it into array or easy calculations later

        # list of points with probability less than or equal to the threshold
        epsilon_list = [item for item in prob_cv if item <= prob_cv.mean()]  
    

        # The F-score, also called the F1-score, is a measure of a model's accuracy on a dataset
        # Calculate the f-score:
        f_score_list = []

        scoreCalculator = ScoreCalculator()

        for item in epsilon_list:
            f_score_list.append(scoreCalculator.f_score(item, prob_cv, array_label))

        max_fscore_index = np.array(f_score_list).argmax()
        max_fscore_epsilon = epsilon_list[max_fscore_index]

        # This list will contain 0 or 1 value and that will tell if a data point is outlier or not
        # if there is 0 at index i then that means the point is normal
        # if there is 1 at index i then that means the point is anomalous
        anomaly_list = []  

        for i in range(0, len(file_x)):
            if prob_x[i] <= max_fscore_epsilon:
                anomaly_list.append(1)
            else:
                anomaly_list.append(0)


        file_x['label'] = np.array(anomaly_list)

        # plotting graph with anomalies
        plt.figure()

        for index, row in file_x.iterrows():
            if int(row['label']) == 1:
                plt.scatter(row[0], row[1], color='red')
            else:
                plt.scatter(row[0], row[1], color='blue')
        plt.show()



    def __probability(self, file: DataFrame) -> Series:
        """The method first calculates the probability  of all the data points
        and then their variance in bell curve. Based on these values it calculates the 
        threshold probability value which is used to determine if a data point is 
        an outlier or not.

        Args:
            file (DataFrame): The excel file in panada DataFrame format.

        Returns:
            Series: <class 'pandas.core.series.Series'> 
            The series contains probability values of each point int the file.
        
        Useful links:
            https://towardsdatascience.com/a-complete-anomaly-detection-algorithm-from-scratch-in-python-step-by-step-guide-e1daf870336e
        """

        # total number of rows or data points
        total_records = len(file)
        # sum of all the values n first column and second column separately
        sum = np.sum(file, axis=0)
        # mean of both the columns
        mean = sum/total_records

        variance_denominator = np.sum((file - mean)**2, axis=0)
        # variance
        variance_square = variance_denominator/mean
        # diagonal values of the variance
        variance_diag = np.diag(variance_square)
        # compute the determinant
        variance_sq_determinant = np.linalg.det(variance_diag)

        mean_len = len(mean)
        X = file - mean

        # Guassian's formula
        prob = 1/((2*np.pi)**(mean_len/2)*(variance_sq_determinant**0.5))*np.exp(-0.5* np.sum(X @ np.linalg.pinv(variance_diag) * X, axis=1))
        
        return prob

