# what is streamlit?
Streamlit turns data scripts into shareable web apps in minutes.
All in pure Python. No front‑end experience required.It helps us create web apps for data science and machine learning in a short time. 
It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

# Prequest
- you have to install python which has pip, how to install pip can be seen [here](https://pip.pypa.io/en/stable/installation/).
- run the installation on the file [requirements.txt](requirements.txt)
- ```commandline 
  pip install -r /path/to/requirements.txt
    ```
  
# How to run this code
- after you have installed pip, next is to run the program by running the following command:
- ```commandline
    streamlit run summary.py
    ```
- then you will be directed to the socket which is opened to show the frontend view
- if you want to share publicly, try reading [this documentation](https://towardsdatascience.com/deploy-a-public-streamlit-web-app-for-free-heres-how-bf56d46b2abe)
- if you want to change data(path) then change on 27th line.
- 
# def fuction
- `st_wirete_accuracy()`. This function is created to call values from Accuracy, Recall, Precision and F-measure.
- `button_display()`. This function is to display the buttons on each page
- `logo_()`. shows where the logo and logo creation is in the center
- `confusion_matrix_plot(x,y)`. to display the confusion matrix on each model
- `smote()`. this function will take a little time to run as it is required in calling all three models.
- `grafik_actual_vs_predict()`.This function will call the predicted result which will be compared with the actual situation