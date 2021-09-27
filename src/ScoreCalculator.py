from typing import Tuple
import numpy
from pandas.core.series import Series


class ScoreCalculator:

    def f_score(self, epsilon: float, prob_x: Series, file_cv: numpy.ndarray) -> float:
        """This method calculates the f-score of the model to determine it's accuracy.
        It calls private method (__calc_tf_fp_fn) to get the tf, fp and fn score and use those value to 
        calculate precision and recall value that are used to calculate f-score.

        For more information on f-score:
        https://towardsdatascience.com/a-complete-understanding-of-precision-recall-and-f-score-concepts-23dc44defef6

        Args:
            epsilon (float): [description]
            prob_x (Series): [description]
            file_cv (numpy.ndarray): [description]

        Returns:
            float: f-score value of the model
        """

        
        # To calculate the f-score we first need to calculate truePositive, falsePositive, falsePositive
        # values of the list
        truePositive, falsePositive, falseNegative = self.__calc_tf_fp_fn(prob_x, epsilon, file_cv)

        # print(truePositive, falsePositive, falseNegative)
        precision_value = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)

        f_score_value = (2*precision_value*recall) / (precision_value + recall)

        return f_score_value

    def __calc_tf_fp_fn(self, epsilon: float, prob_x: Series, file_cv: numpy.ndarray) -> Tuple [float, float, float]:  
        """Calculates true positives, false positives and false negatives of the model provided.

        Args:
            epsilon (float): threshold probability value
            prob_x (Series): probability value of each data point
            file_cv (numpy.ndarray): Cross-validation array

        Returns:
            Tuple [float, float, float]: This method returns true positives, false positives and false negatives
            values of the model respectively.
        """
        

        truePositive, falsePositive, falseNegative = 0, 0, 0

        for i in range(0, len(file_cv)):
            # if the probability of item at index 'i' is less than the epsilon and it has is anomalous item then
            # increase the true positives by one
            # this means that our training model was able to detect anomalous; which is good
            if prob_x[i] <= epsilon and file_cv[i][0] == 1:
                truePositive += 1

            # if the probability of item at index 'i' is less than the epsilon  but it is not an anomalous point then
            # that is not good, this is where my training model is wrong which means it is flagging an item which is
            # non anomalous as anomalous
            elif prob_x[i] <= epsilon and file_cv[i][0] == 0:
                falsePositive += 1

            # This basically means that my training model is says a point is not anomalous but it should be
            elif prob_x[i] > epsilon and file_cv[i][0] == 1:
                falseNegative += 1

        return truePositive, falsePositive, falseNegative
