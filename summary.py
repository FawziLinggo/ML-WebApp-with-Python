from st_aggrid import AgGrid
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# Model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer


data = pd.read_csv('/home/adi/PycharmProjects/streamlit-testing/data/inidataset.csv')

st.markdown("# Summary page 🎈")
st.sidebar.markdown("# Main page 🎈")


## UKT Minimum label
# st.area_chart(pd.value_counts(data['UKT (Minimum) label']))
st.bar_chart(pd.value_counts(data['UKT (Minimum) label']))

X = data.iloc[:,2:-1].values
y = data.iloc[:,-1].values

scaler2 = StandardScaler()
X = scaler2.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

counter = Counter(y_train)
print(counter)
# print(data.describe())
st.write("""%s""" %counter)


st.write("""
    # DataFrame Train
""")
st.line_chart(pd.DataFrame(X_train))

st.write("""
    # DataFrame Test
""")
st.line_chart(pd.DataFrame(X_test))


# compute required values
scaler = StandardScaler().fit(X_train)
train_sc = scaler.transform(X_train)
test_sc = scaler.transform(X_test)

st.write("""Data Describe""")
st.write(data.describe())

def svm():
    svc = SVC()
    svc.fit(train_sc, y_train)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, train_sc, y_train, cv=kfold, scoring=scoring)


    hasil = pd.DataFrame(y_test, columns=['Aktual'])
    # hasil['Prediksi'] = pd.DataFrame(pr)
    #

    st.write("""
        # Table Predict SVM
     """)
    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))
    # st.table(hasil)
    # AgGrid(hasil)

def mlp_classifier():
    mlp = MLPClassifier()
    mlp.fit(train_sc, y_train)
    mlp_pred = mlp.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    mlp_clf = MLPClassifier()
    scores = cross_validate(mlp_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    st.write("""
        # Table Predict MLP Classifier
     """)

    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

def random_forest():
    rf = RandomForestClassifier()
    rf.fit(train_sc, y_train)
    rf_pred = rf.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    rf_clf = RandomForestClassifier()
    scores = cross_validate(rf_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    st.write("""
        # Table Predict Random Forest
     """)

    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

def smote():
    oversample = SMOTE()
    X_train_res, y_train_res = oversample.fit_resample(train_sc, y_train)
    st.bar_chart(pd.value_counts(y_train_res))