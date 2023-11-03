import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

with st.form(key ='Form1'):
    uploaded_pdf = st.file_uploader("Load pdf: ", type=['pdf'])

    if uploaded_pdf is not None:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        #st.write(text) 
        doc.close()
    #----------
    #if uploaded_file is not None:
    #    df = extract_data(uploaded_file)
    #    st.write(df)
    #load position info
    
    position_title=st.text_input('job position')
    description=st.text_area('job description')
    
    submit_code = st.form_submit_button(label ="Execute")

if submit_code:
    # Load the key from a file
    api_key = st.secrets.key#open(st.secrets.key, 'r').read().strip('\n')
    assert api_key.startswith('sk-'), 'Error loding the API key. OpenAI API Keys start with "sk-".'
    openai.api_key = api_key
    
    
    
    #system role
    role="eres un reclutador de RRHH experto, que puede analizar en detalle el curriculum, \
        entregar match con la posición a comparar, \
        y dar feedback para mejorarlo"
    
    #instructions
    instr_1="Entregar un score de 1 a 100 respecto al match de las skills del cargo vs curriculum. Sólo entregar el número del score, el cual será guardado en una variable float"
    
    #prompt with a chat model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens= 200,
        messages=[{"role": "system",
                   "content": role},
                  {"role": "user",
                   "content": instr_1 + "curriculum:"+text + "Cargo a postular:"+ position_title + "Descripción cargo:"+description}]
    )
    numbers = np.array(response["choices"][0]["message"]["content"])
    result = numbers[np.char.isnumeric(numbers)].astype(int)

    #fig, ax = plt.subplots()
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = result,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match Scoring"}))

    st.pyplot(fig)
    st.text(result)
