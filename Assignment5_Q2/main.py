#-------------------------------------------------------------------------
# AUTHOR: Chanrady Ho
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df
max_sil_co = 0
cur_sil_co = 0
silhouette_coefficient_df = []

#run kmeans testing different k values from 2 until 20 clusters
for k in range(2, 21):
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     cur_sil_co = silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficient_df.append(cur_sil_co)
     if cur_sil_co > max_sil_co:
          max_sil_co = cur_sil_co
          max_k = k
print("k maximizes the silhouette coefficient: " + str(max_k))

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
plt.scatter(silhouette_coefficient_df, k)
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
test_df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(test_df.values).reshape(1, test_df.shape[0])[0]

#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())