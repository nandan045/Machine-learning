#Random Forest

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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predict the test results
y_pred = classifier.predict(X_test)

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

