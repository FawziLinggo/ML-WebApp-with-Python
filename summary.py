import altair
import streamlit as st
## Important
st.set_option('deprecation.showPyplotGlobalUse', False)
from time import time
import warnings
warnings.filterwarnings("ignore")

import pandas  as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import webbrowser

# Model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

data = pd.read_csv('/home/adi/PycharmProjects/streamlit-testing/data/inidataset-1.csv')

## UKT Minimum label
data = data[data["UKT (Minimum) label"] != 0]
X = data.drop(columns=["program_studi", "get_ukt", "UKT (Minimum) label"]).values
y = data["UKT (Minimum) label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler2 = StandardScaler()
X = scaler2.fit_transform(X)

counter = Counter(y_train)
# compute required values
scaler = StandardScaler().fit(X_train)
train_sc = scaler.transform(X_train)
test_sc = scaler.transform(X_test)

def svm():
    button_display()
    svc = SVC()
    svc.fit(train_sc, y_train)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, train_sc, y_train, cv=kfold, scoring=scoring)


    hasil = pd.DataFrame(y_test)
    hasil['Prediksi'] = pd.DataFrame(svc_pred)
    st.line_chart(hasil)

    st.write("""
        # Table Predict SVM
     """)
    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

    st.subheader("Confusion Matrix SVM")
    confusion_matrix_plot(y_test, svc_pred)




def mlp_classifier():
    button_display()
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

    st.subheader("Confusion Matrix MLP")
    confusion_matrix_plot(y_test, mlp_pred)

def random_forest():
    button_display()
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

    st.subheader("Confusion Matrix Random Forest")
    confusion_matrix_plot(y_test, rf_pred)



def smote():
    button_display()
    oversample = SMOTE()
    X_train_res, y_train_res = oversample.fit_resample(train_sc, y_train)
    st.bar_chart(pd.value_counts(y_train_res))

    time_svm = time()
    svc = SVC()
    svc.fit(X_train_res, y_train_res)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_svm_calculate= time()-time_svm

    st.write("""
        # Predict SVM SMOTE
     """)

    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

    time_mlp = time()
    mlp = MLPClassifier()
    mlp.fit(X_train_res, y_train_res)
    mlp_pred = mlp.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    mlp_clf = MLPClassifier()
    scores = cross_validate(mlp_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_mlp_calculate= time()-time_mlp


    st.write("""
        # Predict MLP Classifier SMOTE
     """)

    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

    time_rf = time()
    rf = RandomForestClassifier()
    rf.fit(X_train_res, y_train_res)
    rf_pred = rf.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    rf_clf = RandomForestClassifier()
    scores = cross_validate(rf_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_rf_calculate= time()-time_rf


    st.write("""
        # Table Predict Random Forest
     """)

    st.write(""" Mean Accuracy: %f """ %np.mean(scores['test_accuracy']))
    st.write(""" Mean Recall: %f """ %np.mean(scores['test_recall_macro']))
    st.write(""" Mean Precision: %f  """ %np.mean(scores['test_precision_macro']))
    st.write(""" Mean F-measure: %f """ %np.mean(scores['test_f1_macro']))

    data_time_calculate =pd.DataFrame(
        {'SVM' : [time_svm_calculate],
        'MLP' : [time_mlp_calculate],
        'Random Forest' : [time_rf_calculate]}
    )

    # "Time" : [1,2,3],
    # 'Model': ['SVM', 'MLP', 'Random Forest']

    bar_chart = altair.Chart(data_time_calculate).mark_bar().encode(
        y='Time (s):Q',
        x='Model:O',
    )
    st.bar_chart(data_time_calculate.loc[0],use_container_width=True)

def confusion_matrix_plot(x,y):
    conf_mat = confusion_matrix(x, y)
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1", "2", "3", "4", "5", "6"]).plot()
    st.pyplot()
    return x,y

def st_wirete_accuracy(a,b,c,d):
    st.write(""" Mean Accuracy: %f """ %a)
    st.write(""" Mean Recall: %f """ %b)
    st.write(""" Mean Precision: %f  """ %c)
    st.write(""" Mean F-measure: %f """ %d)

def button_display():
    url = 'http://localhost:8503/'
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        button1 = st.button('Home')
    with col2:
        button2 = st.button('Random Forest')
    with col3:
        button3 = st.button('MLP Classifier')
    with col4:
        button4 = st.button('SVM')
    with col5:
        button5 = st.button('Smote')

    if button1:
        webbrowser.open(url)
    if button2:
        webbrowser.open(url + 'model_randomForest')
    if button3:
        webbrowser.open(url + 'model_mlp')
    if button4:
        webbrowser.open(url + 'model_svm')
    if button5:
        webbrowser.open(url + 'smote')

if __name__ == '__main__':

    st.markdown("# Penentuan Klasifikasi UKT Berbasis "
                "*Machine Learning*")
    button_display()

    st.write("""Data Describe""")
    st.write(data.describe())

    st.bar_chart(pd.value_counts(data['UKT (Minimum) label']))

    # print(data.describe())
    st.write("""%s""" % counter)

    st.write("""
        # DataFrame Train
    """)
    # st.line_chart(pd.DataFrame(X_train))
    st.bar_chart(pd.DataFrame(X_train))

    st.write("""
        # DataFrame Test
    """)
    # st.line_chart(pd.DataFrame(X_test))
    st.bar_chart(pd.DataFrame(X_test))


