import os
import sys
import pandas as pd
import statistics
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib
from utils import read_files

#This is the path to the data
path = sys.argv[-1]

categories_list = next(os.walk(path))[1]
df_list = []
for category in categories_list:
    df_list.append(read_files(category))
df = pd.concat(df_list)
df = df.reset_index(drop=True)

X = df["Text"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

pipeline_tranform_data = joblib.load("pipeline_tranform_data.pkl")
X_train_vector = pd.DataFrame(pipeline_tranform_data.fit_transform(X_train))
X_test_vector = pd.DataFrame(pipeline_tranform_data.fit_transform(X_test))


C = np.logspace(0, 2, 5)
gamma = np.logspace(0, 2, 5)

random_grid = {'C': C, 'gamma': gamma}

svc = SVC()
best_model = RandomizedSearchCV(
                estimator = svc,
                param_distributions = random_grid,
                n_iter=100,
                cv=4,
                scoring='accuracy',
                verbose=5,
                random_state=42,
                n_jobs=-1)

best_model.fit(X_train_vector, y_train)
print(best_model.best_params_)

print("Scores for the Train Dataset: ")
y_train_pred = best_model.predict(X_train_vector)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: %.2f%%" % (accuracy_train * 100.0))


print("- - - - - - - - - - ")

print("Scores for the Test Dataset: ")
y_test_pred = best_model.predict(X_test_vector)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy_test * 100.0))

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print(accuracy_score(y_test, y_test_pred))

joblib.dump(best_model, "model.pkl")

