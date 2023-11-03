import streamlit as st
import pandas as pd
import openai
import os

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
#if uploaded_file is not None:
#    df = extract_data(uploaded_file)
#    st.write(df)

# Load the key from a file
api_key = open(st.secrets.key, 'r').read().strip('\n')
assert api_key.startswith('sk-'), 'Error loding the API key. OpenAI API Keys start with "sk-".'
openai.api_key = api_key

#load position info
position_title=st.text_input('job position')
description=st.text_input('job description')

submit_code = st.form_submit_button(label ="Execute")

#system role
role="eres un reclutador de RRHH experto, que puede analizar en detalle el curriculum, \
    entregar match con la posición a comparar, \
    y dar feedback para mejorarlo"

#instructions
instr_1="Entregar un score de 1 a 100 respecto al match de las skills del cargo vs curriculum.#\
       # Además entregar 5 principales razones del score."
instr_2="Entregar feedback con 3 principales fortalezas y 3 principales debilidades\
    del curriculum versus la descripción del cargo."
instr_3="Entregar listado de 5 preguntas que pueden preguntar en el contexto de la empresa\
    y la descripción del cargo"

#prompt with a chat model
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system",
               "content": role},
              {"role": "user",
               "content": instr_1 + "curriculum:"+uploaded_file + "Cargo a postular:"+
                          position_title + "Descripción cargo:"+description}]
)
if submit_code:
    st.text(response["choices"][0]["message"]["content"])
