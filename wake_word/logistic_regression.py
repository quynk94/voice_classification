# coding= UTF-8

import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import config

# Load data from numpy file
X = np.load('extracted_data/feat.npy')
y = np.load('extracted_data/label.npy').ravel()

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)


# Logistic Regression
print('fitting...')
target_names = config.target_names()
params = {"C": [0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(
    solver="lbfgs", multi_class="auto", max_iter=300), params, cv=3, n_jobs=-1)
model.fit(X_train, y_train)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
preds = model.predict(X_test)
print(classification_report(y_test, preds, target_names=target_names))
filename = 'models/logistic.pkl'
pickle.dump(model, open(filename, 'wb'))

# predict_feat_path = 'extracted_data/predict_feat.npy'
# predict_filenames = 'extracted_data/predict_filenames.npy'
# filenames = np.load(predict_filenames)
# X_predict = np.load(predict_feat_path)
# pred = model.predict(X_predict)
# for pair in list(zip(filenames, pred)):
#     print(pair)
