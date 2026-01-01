import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jbl

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve

from sklearn.metrics import ConfusionMatrixDisplay

from src.custom_threshold_classifier import ThresholdClassifier

train = pd.read_csv('Data/Paitients_Files_Train.csv')
test = pd.read_csv('Data/Paitients_Files_Test.csv')

train['Sepssis'] = train['Sepssis'].replace({'Negative': 0, 'Positive': 1}).astype(int)

X = train.drop(['ID','Sepssis'], axis=1)
y = train['Sepssis']

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)

tuned_model = Pipeline(
    steps = [ ('imputer', SimpleImputer(strategy='mean') ),
              ('scaler', StandardScaler() ),
              ('balanced_sampling', RandomUnderSampler()),
              ('model', ThresholdClassifier(model=LogisticRegression(), threshold=0.45))
    ]
)
tuned_model.fit(train_x, train_y)

predict_train = tuned_model.predict(train_x)
predict_test = tuned_model.predict(test_x)

print("Training Confusion Matrix : \n", confusion_matrix(train_y,predict_train))
print("Test Confusion Matrix : \n", confusion_matrix(test_y,predict_test))

print("Training Classification Report : \n", classification_report(train_y,predict_train))
print("Test Classification Report : \n", classification_report(test_y,predict_test))

final_prediction = tuned_model.predict(test.drop('ID', axis=1))
print("Final Prediction For Real Test Data")
print(final_prediction)


jbl.dump(tuned_model,'Models/Model_Balanced_45_Threshold.pkl')


