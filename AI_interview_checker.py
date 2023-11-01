import streamlit as st
import pandas as pd
import numpy as np


uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    df = extract_data(uploaded_file)

st.dataframe(df)

