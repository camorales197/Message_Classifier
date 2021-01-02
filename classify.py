
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



model = sys.argv[1]
print("The model is ", model)
files_list = sys.argv[2:]
print("The path to the file/s is/are: ", files)

model = joblib.load(model)
pipeline_tranform_data = joblib.load("pipeline_tranform_data.pkl")

files_content_list = []

for file_path in files_list:
    content = open(file_path, 'r', errors='ignore').read()
    files_content_list.append(content)
    df_files_content = pd.DataFrame(files_content_list, columns=['Text'])


X = df["Text"]
X_vector = pd.DataFrame(pipeline_tranform_data.fit_transform(X))

y = model.predict(X_vector)

solution = {'files': files_list, 'predictions': y}
solution_df = pd.DataFrame(solution)
print(solution_df)
