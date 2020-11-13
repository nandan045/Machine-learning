import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score)

################################### DATA PREPROCESSING #############################################

# Importing the dataset
dataset = pd.read_csv('data.csv', sep = '|')
X = dataset.drop(['Name', 'md5', 'legitimate'], axis = 1).values
y = dataset['legitimate'].values

# Tree-based feature selection:
from sklearn.feature_selection import SelectFromModel
import sklearn.ensemble as ske
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, dataset.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))
features = []
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(dataset.columns[2+f])
    
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################################### PRINTING SCORES ################################################

def print_performance(y_pred , y_test):
    y_pred = y_pred.round()
    print("Accuracy score :" ,accuracy_score(y_test, y_pred))
    print("Precision score : ",precision_score(y_test ,y_pred,average='weighted'))
    print("Recall score : ",recall_score(y_test , y_pred ,average='weighted'))
    print("F1 score : ",f1_score(y_test , y_pred, average="weighted"))  # average = default is when the label is binary class, since we have multivalued class we have to use weighted
    # print("================CONFUSION MATRIX===============")
    conf_matrix = confusion_matrix(y_test, y_pred)
    # print(conf_matrix)
    print(pd.crosstab(y_test, y_pred , rownames = ['TRUE'] , colnames = ['PREDICTED'] , margins = True))

################################# SIMPLE KNN #######################################################

classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(" #############################    SIMPLE KNN    #################################### ")
print_performance(y_pred, y_test)

####################################################################################################

################################  RANDOM FOREST  ###################################################

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predict the test results
y_pred = classifier.predict(X_test)
print(" ############################    RANDOM FOREST   #################################### ")
print_performance(y_pred, y_test)

#####################################################################################################

################################  LOGISTIC REGRESSION  ##############################################

clfLog = LogisticRegression()
clfLog.fit(X_train,y_train)
y_pred = clfLog.predict(X_test)
print(" ############################    LOGISTIC REGRESSION   #################################### ")
print_performance(y_pred, y_test)

#####################################################################################################

##################################  DECISION TREE  ##################################################

clfDT = DecisionTreeRegressor()
clfDT.fit(X_train,y_train)
y_pred = clfDT.predict(X_test)
print(" ###########################  DECISION TREE #################################### ")
print_performance(y_pred, y_test)

#####################################################################################################

###################################  SVM ############################################################

svm1 = svm.SVC()
svm1.fit(X_train,y_train)
y_pred = svm1.predict(X_test)
print(" ##############################  SVM  ##########################################" )
print_performance(y_pred, y_test)

#####################################################################################################

######################################  NAIVE BYES  #################################################

clfNB = GaussianNB()
clfNB.fit(X_train,y_train)
y_pred = clfNB.predict(X_test)
print(" ################################ NAIVE BYES ##################################### ")
print_performance(y_pred, y_test)

#####################################################################################################


