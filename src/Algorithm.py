def algorithm(self):
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Excel file used in this example has 3 sheets (X, Xval & y)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Read the file using panda library function

    # Read the main sheet i.e. 
    file_x = pd.read_excel('ex8data1.xlsx', sheet_name='X', header=None)
    # print(file_x)
    prob_x = self.calc_probability(file_x)

    # prob_x.describe()
    print("****")

    # Read the cross-validation data
    # Credit: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiTmOzFh8PtAhUWuZ4KHRioAcMQFjADegQICBAC&url=https%3A%2F%2Fmachinelearningmastery.com%2Fk-fold-cross-validation%2F&usg=AOvVaw3dq5uCuxWnEEzbZGBP5vK2
    # Why do we need cross validation data?
    # Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning
    # model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to
    # perform in general when used to make predictions on data not used during the training of the model

    # here we are going to ead the cross validation data to calculate the probability
    file_cv = pd.read_excel('ex8data1.xlsx', sheet_name='Xval', header=None)
    prob_cv = self.calc_probability(file_cv)

    # This will give some statistical figures on the data that we have
    # print(prob_cv.describe())
    print("****")


    # Read labels file
    # Labels file is used as the ground truth to check if the training model is working good or not
    file_label = pd.read_excel('ex8data1.xlsx', sheet_name='y', header=None)
    array_label = np.array(file_label)  # converting it into array or easy calculations later


    # # # # # #
    # if you want to change the parameters of probability to see how the plot changes based on value
    # multiply prob_cv.mean() by a factor and it will change the plot
    # # # # # #

    # list of with probability less than or equal to the threshold
    epsilon_list = [item for item in prob_cv if item <= prob_cv.mean()]  # list of items with value probability
    # less than mean value of all probability

    print(prob_cv.mean())
    print(len(epsilon_list))


    # The F-score, also called the F1-score, is a measure of a model's accuracy on a dataset
    # https://deepai.org/machine-learning-glossary-and-terms/f-score

    # here we will calculate the f score list
    f_score_list = []  # f-score value of every item
    for item in epsilon_list:
        f_score_list.append(self.f_score(item, prob_cv, array_label))

    # print(len(f_score_list))

    max_fscore_index = np.array(f_score_list).argmax()

    max_fscore_epsilon = epsilon_list[max_fscore_index]

    anomaly_List = []  # This list will contain 0 or 1 value and that will tell if an item is anomalous or not
    # if there is 0 at index i then that means the point is normal
    # if there is 1 at index i then that means the point is anomalous

    for i in range(0, len(file_x)):
        if prob_x[i] <= max_fscore_epsilon:
            anomaly_List.append(1)
        else:
            anomaly_List.append(0)


    file_x['label'] = np.array(anomaly_List)

    # plotting graph with anomalies
    plt.figure()
    # plt.scatter(prob_x[0], prob_x[1])

    print(file_x)
    for index, row in file_x.iterrows():
        # print(row['label'])
        if int(row['label']) == 1:
            # print("******\n", row, "\n")
            plt.scatter(row[0], row[1], color='red')
        else:
            plt.scatter(row[0], row[1], color='blue')
    plt.show()
    