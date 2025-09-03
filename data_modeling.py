import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

def classification_report_output(name, y_actual, y_pred, target_names=None):

    print(classification_report(y_true=y_actual, y_pred=y_pred, 
                                digits=4, target_names=target_names))

    f = open('Ensemble_Model_Classification_Report.txt', 'a')
    f.writelines(f"########## {name} ##########\n")
    f.writelines(f"{classification_report(y_true=y_actual, y_pred=y_pred, digits=4, target_names=target_names)}\n")
    f.writelines(f"############################\n")
    f.close()

df = pd.read_csv('aggregated_cleaned.csv')

x = df.drop('TARGET', axis=1)
y = df['TARGET']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

model=RandomForestClassifier().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="Random Classifier",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")

model=GradientBoostingClassifier().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="GradientBoostingClassifier",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")

model=AdaBoostClassifier().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="AdaBoostClassifier",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")

model=SVC().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="SVC",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")

model=DecisionTreeClassifier().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="DecisionTreeClassifier",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")

model=XGBClassifier().fit(x_train,y_train)
y_predict=model.predict(x_test)


print("########## Ensemble Model ##########")
classification_report_output(name="XGBClassifier",
                            y_actual=y_test,
                            y_pred=y_predict, target_names=["Unworn","Worn"])
print("############################")





# print("########## KNN ##########")
# classification_report_output(name="KNN",
#                             y_actual=y,
#                             y_pred=model_knn.predict(X), target_names=["Unworn","Worn"])
# print("############################")

# print("########## Random Forest ##########")
# classification_report_output(name="RF",
#                             y_actual=y,
#                             y_pred=model_rf.predict(X), target_names=["Unworn","Worn"])
# print("############################")

# print("########## LightGBM ##########")
# classification_report_output(name="LightGBM",
#                             y_actual=y,
#                             y_pred=model_lightgmb.predict(X), target_names=["Unworn","Worn"])
# print("############################")