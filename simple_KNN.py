# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

# Importing the dataset
dataset = pd.read_csv('data.csv', sep = '|')
X = dataset.drop(['Name', 'md5', 'legitimate'], axis = 1).values
y = dataset['legitimate'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

def print_performance(y_pred , y_test):
    print("Accuracy score :" ,accuracy_score(y_test, y_pred))
    print("Precision score : ",precision_score(y_test ,y_pred,average='weighted'))
    print("Recall score : ",recall_score(y_test , y_pred ,average='weighted'))
    print("F1 score : ",f1_score(y_test , y_pred, average="weighted"))  # average = default is when the label is binary class, since we have multivalued class we have to use weighted
    # print("================CONFUSION MATRIX===============")
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    print(pd.crosstab(y_test, y_pred , rownames = ['TRUE'] , colnames = ['PREDICTED'] , margins = True))

print_performance(y_pred, y_test)

#from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
#myList = list(range(1,50))
#
# subsetting just the odd ones
#neighbors = filter(lambda x: x % 2 != 0, myList)
#neighbors = list(range(1,50))
# empty list that will hold cv scores
#cv_scores = []
#
# perform 20-fold cross validation
#for k in neighbors:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, X_train, y_train, cv=20, scoring='accuracy')
#    cv_scores.append(scores.mean())
#
# changing to misclassification error
#MSE = [1 - x for x in cv_scores]
#MSE_list = np.array(MSE)
#neighbors_list = np.array(neighbors)
# determining best k
#optimal_k = neighbors[MSE_list.tolist().index(min(MSE_list))]
#print ("The optimal number of neighbors is %d" % optimal_k)
#
# plot misclassification error vs k
#plt.plot(neighbors_list, mse_list)
#plt.xlabel('number of neighbors k')
#plt.ylabel('misclassification error')
#plt.show()
