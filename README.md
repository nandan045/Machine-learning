# Machine-learning
The implementation of the project “malware detection using machine learning” is been done as per the SRS document and the design document. We have used several machine learning classification algorithms to detect and predict malwares. 

The software was to be implemented using python and MySql. MySql was supposed to store the dataset, whereas the python programming language was used for machine learning. 

Initially several machine learning algorithms were executed and the prediction score was documented. Later, the feature selection feature was added to the machine learning pipeline by which the prediction score was increased with the speed of the execution.

•	The extract_features.py script extracts features from the raw executable files.
•	Feature_selection.py script selects 10 important features out of many features (56) and returns the prediction scores of all algorithms at a time.
•	Simple-KNN.py script returns the prediction score of KNN algorithm.
•	Random forest.py script returns the prediction score of random forest algorithm.
•	SVM.py script returns the prediction score of SVM algorithm.
•	Logical regression.py script returns the prediction score of logical regression algorithm.
•	Decision tree.py script returns the prediction score of decision tree algorithm.
•	Naïve bayes script returns the prediction score of naïve bayes algorithm.

numpy, pandas and sklearn python modules are required for this scripts. Python 3+ version is advisable.
