class ScoreCalculator:

    def f_score(self, epsilon, prob_x, file_cv):

        # For more information on f-score:
        # credit: https://towardsdatascience.com/a-complete-understanding-of-precision-recall-and-f-score-concepts-23dc44defef6

        # To calculate the f-score we first need to calculate truePositive, falsePositive, falsePositive
        # values of the list
        truePositive, falsePositive, falseNegative = self.__calc_tf_fp_fn(prob_x, epsilon, file_cv)

        # print(truePositive, falsePositive, falseNegative)
        precision_value = truePositive / (truePositive + falsePositive)
        recall = truePositive / (truePositive + falseNegative)

        f_score_value = (2*precision_value*recall) / (precision_value + recall)

        return f_score_value

    def __calc_tf_fp_fn(self, prob_x, epsilon, file_cv):  # Calculates true positives, false positives and false negatives

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
