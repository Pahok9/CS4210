# -------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: Naive Bayes on weather prediction
# SPECIFICATION: Using Naive Bayes to predict the weather given some data from past results
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
# --> add your Python code here
db = []
with open('weather_training.csv', "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row[1:6])

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
X = []
features = {0: {'Sunny': 1, 'Overcast': 2, 'Rain': 3},
            1: {'Cool': 1, 'Mild': 2, 'Hot': 3},
            2: {'Normal': 1, 'High': 2},
            3: {'Weak': 1, 'Strong': 2},
            4: {'Yes': 1, 'No': 2}}

for i in range(len(db)):
    instance = [features[0][db[i][0]], features[1][db[i][1]], features[2][db[i][2]], features[3][db[i][3]]]
    X.append(instance)

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
Y = []
for i in range(len(db)):
    instance = features[4][db[i][4]]
    Y.append(instance)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# --> add your Python code here
db = []
with open('weather_test.csv', "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row[0:5])

db_test = []
for i in range(len(db)):
    instance = [features[0][db[i][1]], features[1][db[i][2]], features[2][db[i][3]], features[3][db[i][4]]]
    db_test.append(instance)

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(
    15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

for i, instance in enumerate(db):
    # use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
    # --> add your Python code here
    class_predicted = clf.predict_proba([db_test[i]])[0]
    if class_predicted[0] >= 0.75:
        play_tennis = "yes"
        print(db[i][0].ljust(15), db[i][1].ljust(15), db[i][2].ljust(11), db[i][3].ljust(15),
              db[i][4].ljust(15), play_tennis.ljust(15), str(round(class_predicted[0], 2)).ljust(15))
    elif class_predicted[1] >= 0.75:
        play_tennis = "No"
        print(db[i][0].ljust(15), db[i][1].ljust(15), db[i][2].ljust(11), db[i][3].ljust(15),
              db[i][4].ljust(15), play_tennis.ljust(15), str(round(class_predicted[1], 2)).ljust(15))
