import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import webbrowser

# Model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# Important
from time import time
import warnings

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

# path to fetch data (Linux)
data = pd.read_csv('data/inidataset-1.csv')
path_logo='data/poltek-removebg-preview.png'

# UKT Minimum labels
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


def grafik_actual_vs_predict(pred):

    # ----------------------------------- EXPLANATION -----------------------------------
    #     the grafik_actual_vs_predict() function is a function that will
    #     retrieve the value from the y_test data. with streamlit using
    #     subheader to show sub header, then y_test result is actual and added prediction
    #     dataframe. which will return the value of the predicted result. and the final
    #     result will be displayed in the form of a line chart using streamlit.
    #
    #     You can see these docs in this link:
    #     https://docs.streamlit.io/library/api-reference/text/st.subheader
    #     https://docs.streamlit.io/library/api-reference/charts/st.line_chart
    #
    #     besides that you can also use the area chart (area_chart) in streamlit,
    #     see the documentation:
    #     https://docs.streamlit.io/library/api-reference/charts/st.area_chart
    # ---------------------------------------------------------------------------------

    st.subheader("Prediksi Testing [Actual vs Prediction]")
    hasil= pd.DataFrame(y_test)
    hasil['Prediksi'] = pd.DataFrame(pred)
    st.line_chart(hasil)

def confusion_matrix_plot(x,y):

    # ----------------------------------- EXPLANATION -----------------------------------
    #     this is a function to display an image from
    #     the confusion matrix with the algorithm that has been built.
    #     streamlit supports to display images with pytlot to Display
    #     matplotlib.pyplot images.
    #     by returning the values of x and y. where x is y_test
    #     and y is prediction result (pred)
    #
    #     To view streamlit documentation on Pyplot, see the following link:
    #     https://docs.streamlit.io/library/api-reference/charts/st.pyplot
    # ---------------------------------------------------------------------------------

    conf_mat = confusion_matrix(x, y)
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1", "2", "3", "4", "5", "6"]).plot()
    st.pyplot()
    return x,y

def st_write_accuracy(a,b,c,d):

    # ----------------------------------- EXPLANATION -----------------------------------
    #     Of course, we don't want to repeat the same job over and over again, right?.
    #     function st_write_accuracy() is a function that will return the average result
    #     of Accuracy, Recall, Precision and F-measure. by returning the values a, b, c, d.
    #
    #     :param a: np.mean(scores['test_accuracy'])
    #     :param b: np.mean(scores['test_recall_macro'])
    #     :param c: np.mean(scores['test_precision_macro'])
    #     :param d: np.mean(scores['test_f1_macro'])
    #     :return: a,b,c,d
    #
    #     where to use write function in streamlit. Write arguments to the app.
    #     This is the Swiss Army knife of Streamlit commands: it does different
    #     things depending on what you throw at it. Unlike other Streamlit commands,
    #     write() has some unique properties:
    #
    #     1. You can pass in multiple arguments, all of which will be written.
    #     2. Its behavior depends on the input types as follows.
    #     3. It returns None, so its "slot" in the App cannot be reused.
    #
    #     To view streamlit documentation on write, see the following link:
    #     https://docs.streamlit.io/library/api-reference/write-magic/st.write
    # ---------------------------------------------------------------------------------

    st.write(""" ## Mean Accuracy: *%f* """ %a)
    st.write(""" ## Mean Recall: *%f* """ %b)
    st.write(""" ## Mean Precision: *%f*  """ %c)
    st.write(""" ## Mean F-measure: *%f* """ %d)

def button_display():

    # ----------------------------------- EXPLANATION -----------------------------------
    #     not customizable like CSS in HTML which can use bootstrap, streamlit has
    #     limitation to show button. in the button_display() function we have to
    #     use the column() function in streamlit to make the buttons inline.
    #
    #     where the column() function will divide the section into what we want
    #     where we only make 5 buttons in 1 row means it takes 5 columns.
    #
    #           +----+----+----+----+----+
    #           |col1|col2|col3|col4|col5|
    #           +----+----+----+----+----+
    #
    #     see the documentation here:
    #     https://docs.streamlit.io/library/api-reference/layout/st.columns
    #
    #     After initializing the next column, we create a button with a link that
    #     points to the url: http://localhost with the default port used is 8501.
    #
    #     to be able to open these links, we need a library called webbrowser
    #     which will call the open() function to open links from each class.
    #
    #     streamlit also has a button() function which will return whatever
    #     value we want. To view the documentation, go to the following link :
    #     https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661
    # ---------------------------------------------------------------------------------

    url = 'https://fawzilinggo-ml-webapp-with-python-summary-bt95zb.streamlitapp.com/'
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

def logo_():

    # ----------------------------------- EXPLANATION -----------------------------------
    #     this is what should be simple but quite complicated in streamlit.
    #     streamlit can display an image, but to display an image in the
    #     middle it requires the column() function.
    # ---------------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col2:
        st.image(path_logo)
    with col3:
        st.write(' ')


# ----------------------------------- EXPLANATION -----------------------------------
#     The svm(), mlp_classifier(), and random_forest() functions
#     consist of an algorithm that has been created and adds views for streamlits.
#     which uses :
#     1.  the button_display() function to display the previous algorithm's
#         buttons as well as the main menu.
#     2.  graph_actual_vs_predict() function serves to display the actual vs
#         predicted graph.
#     3.  the st_write_accuracy() function to write the accuracy that has been
#         obtained from the previous coding.
#     4.  and the confusion_matrix_plot() function to display the results of the
#         confusion matrix of each model.
#
#     to see the documentation on how functions work in python, provide the link:
#     https://www.programiz.com/python-programming/function
# ---------------------------------------------------------------------------------
def svm():

    button_display()
    svc = SVC()
    svc.fit(train_sc, y_train)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    grafik_actual_vs_predict(svc_pred)

    st.write("""
        # Table Predict SVM
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']),np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']),np.mean(scores['test_f1_macro']))

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

    grafik_actual_vs_predict(mlp_pred)
    st.write("""
        # Predict MLP Classifier
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

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

    grafik_actual_vs_predict(rf_pred)
    st.write("""
        # Table Predict Random Forest
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    st.subheader("Confusion Matrix Random Forest")
    confusion_matrix_plot(y_test, rf_pred)

# ----------------------------------- EXPLANATION -----------------------------------
#     the smooth() function is the same as the function taken from the previous model,
#     only adding time to calculate the length of each model in training.
#     for that can be seen in the following documentation:
#     https://stackoverflow.com/questions/56203215/how-to-compute-the-time-that-a-machine-learning-model-take-to-classify
#
#     The results will be displayed in a streamlit bar plot. streamlit bar plots use dataframes
#     therefore have to convert model time values into dataframes.
#
#     To view the official documentation on bar charts, see the following links:
#     https://docs.streamlit.io/library/api-reference/charts/st.bar_chart
# ---------------------------------------------------------------------------------
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

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

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

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

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

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    data_time_calculate =pd.DataFrame(
        {'SVM':[time_svm_calculate],
        'MLP':[time_mlp_calculate],
        'Random Forest':[time_rf_calculate]}
    )

    st.bar_chart(data_time_calculate.loc[0],use_container_width=True)



# ----------------------------------- EXPLANATION -----------------------------------
#     in the end we come to the main function. which will return every value
#     from the above function. where the main function displays a summary page
#     of the code we have created. we will also call logo_(),
#     button_display() function (I think it will show on every page :D )
# ---------------------------------------------------------------------------------

if __name__ == '__main__':
    logo_()
    st.markdown("# Penentuan Klasifikasi UKT Berbasis "
                "*Machine Learning*")
    button_display()

    st.write("""Data Describe""")
    st.write(data.describe())

    st.bar_chart(pd.value_counts(data['UKT (Minimum) label']))

    st.write("""%s""" % counter)

    st.write("""
        # DataFrame Train
    """)

    st.bar_chart(pd.DataFrame(X_train))

    st.write("""
        # DataFrame Test
    """)

    st.bar_chart(pd.DataFrame(X_test))


