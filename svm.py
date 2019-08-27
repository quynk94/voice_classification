# coding= UTF-8

import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

# Load data from numpy file
X = np.load('extracted_data/feat.npy')
y = np.load('extracted_data/label.npy').ravel()

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)


# Simple SVM
print('fitting...')
clf = SVC(C=20.0, gamma=0.00001)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("acc=%0.3f" % acc)
filename = 'models/svm.pkl'
pickle.dump(clf, open(filename, 'wb'))

# predict_feat_path = 'extracted_data/predict_feat.npy'
# predict_filenames = 'extracted_data/predict_filenames.npy'
# filenames = np.load(predict_filenames)
# X_predict = np.load(predict_feat_path)
# pred = clf.predict(X_predict)
# for pair in list(zip(filenames, pred)):
#     print(pair)
