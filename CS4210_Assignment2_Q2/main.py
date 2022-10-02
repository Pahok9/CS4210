# -------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: Decision tree on contact lens result
# SPECIFICATION: Using decision tree method to compare the training set and test set to get the accuracy of the results
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

features = {0: {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3},
            1: {'Myope': 1, 'Hypermetrope': 2},
            2: {'Yes': 1, 'No': 2},
            3: {'Normal': 1, 'Reduced': 2},
            4: {'Yes': 1, 'No': 2}}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    accuracy = 0

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    # instance = []
    for i in range(len(dbTraining)):
        instance = [features[0][dbTraining[i][0]], features[1][dbTraining[i][1]], features[2][dbTraining[i][2]],
                    features[3][dbTraining[i][3]]]
        X.append(instance)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> addd your Python code here
    for i in range(len(dbTraining)):
        instance = features[4][dbTraining[i][4]]
        Y.append(instance)

    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = []
        temp = []
        dbY = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:  # skipping the header
                    temp.append(row)
        for j in range(len(temp)):
            instance = [features[0][temp[j][0]], features[1][temp[j][1]], features[2][temp[j][2]],
                        features[3][temp[j][3]]]
            dbTest.append(instance)
            instance = features[4][temp[j][4]]
            dbY.append(instance)

        # for data in dbTest:
        # transform the features of the test instances to numbers following the same strategy done during training,
        # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        # --> add your Python code here
        if i < len(dbTest):
            class_predicted = clf.predict([dbTest[i]])[0]

        # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
        # --> add your Python code here
        if i < len(dbTest) and dbY[i] == class_predicted:
            accuracy += + 1

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        if ds == "contact_lens_training_1.csv":
            final_accuracy1 = accuracy / len(dbTest)
        elif ds == "contact_lens_training_2.csv":
            final_accuracy2 = accuracy / len(dbTest)
        else:
            final_accuracy3 = accuracy / len(dbTest)

# print(final_accuracy2)
    # print the lowest accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    if ds == "contact_lens_training_1.csv":
        print("final accuracy when training on contact_lens_training_1.csv: " + str(final_accuracy1))
    elif ds == "contact_lens_training_2.csv":
        print("final accuracy when training on contact_lens_training_2.csv: " + str(final_accuracy2))
    else:
        print("final accuracy when training on contact_lens_training_3.csv: " + str(final_accuracy3))
