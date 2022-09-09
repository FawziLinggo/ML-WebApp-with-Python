import pandas as pd
import streamlit as st
import pandas as np


st.write("""
testing""")

df = pd.read_csv("data/inidataset.csv")

#
st.line_chart(df)

# Describe
st.line_chart(df.describe())



