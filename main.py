from st_aggrid import AgGrid
import streamlit as st


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


data = pd.read_csv('data/inidataset.csv')



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

    print("Mean Accuracy: ", np.mean(scores['test_accuracy']))
    print("Mean Recall: ", np.mean(scores['test_recall_macro']))
    print("Mean Precision: ", np.mean(scores['test_precision_macro']))
    print("Mean F-measure: ", np.mean(scores['test_f1_macro']))

    hasil = pd.DataFrame(y_test, columns=['Aktual'])
    hasil['Prediksi'] = pd.DataFrame(pr)
    #
    # st.markdown("# Main page 🎈")
    # st.sidebar.markdown("# Main page 🎈")
    st.write("""
        # Table Predict SVM
     """)
    # st.markdown("# Main page 🎈")
    # st.sidebar.markdown("# Main page 🎈")
    # st.table(hasil)
    # AgGrid(hasil)
