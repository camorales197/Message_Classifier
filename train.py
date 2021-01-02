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
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import read_files, TextPreprocessor

#This is the path to the data
path = sys.argv[-1]

#Creating a list with all the categories in the data directory
categories_list = next(os.walk(path))[1]
df_list = []
#Getting the messages from all the categories
for category in categories_list:
    df_list.append(read_files(category))
df = pd.concat(df_list)
df = df.reset_index(drop=True)

#Cleaning the data
pipeline_tranform_data = joblib.load("pipeline_tranform_data.pkl")
Text_Clean = pipeline_tranform_data.fit_transform(df["Text"])

#Splitting the data
X = Text_Clean
y = df["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#Adjusting hyperparameters of the classification model
n_estimators = [1000]
max_features = ['auto']
max_depth = [3, 5, 7]
min_samples_leaf = [2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf
              }

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
                estimator=rf,
                param_distributions=random_grid,
                n_iter=100,
                cv=3,
                scoring='accuracy',
                verbose=5,
                n_jobs=-1)

rf_random.fit(X_train, y_train)
print(rf_random.best_params_)

print("Scores for the Train Dataset: ")
y_train_pred = rf_random.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Accuracy: %.2f%%" % (accuracy_train * 100.0))

print("- - - - - - - - - - ")

print("Scores for the Test Dataset: ")
y_test_pred = rf_random.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy_test * 100.0))

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

joblib.dump(rf_random, "model.pkl")

