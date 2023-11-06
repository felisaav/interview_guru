import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
import json

st.title('AI interview Guru')
tab1, tab2,tab3 = st.tabs(["ResumeRX", "JobMatchMaker", "Read Me"])
with tab1:
    with st.form(key ='Form1'):
        uploaded_pdf = st.file_uploader("Load summary (in pdf format): ", type=['pdf'])
    
        if uploaded_pdf is not None:
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            #st.write(text) 
            doc.close()
        submit_code = st.form_submit_button(label ="Execute")
with tab2:
    with st.form(key ='Form1'):
        uploaded_pdf = st.file_uploader("Load summary (in pdf format): ", type=['pdf'])
    
        if uploaded_pdf is not None:
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            #st.write(text) 
            doc.close()
    
        #load position info
        
        position_title=st.text_input('job position',help='Please record the title of the position you are interested in applying for.')
        description=st.text_area('job description',help='Please record the job description of the position you are interested in applying for.')
        
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
        instr_1="Entregar un score de 1 a 100 respecto al match de las skills del cargo vs curriculum."
        
        instr_2='''Entregar justificación del score con exactamente las 5 principales strengths y 5 principales weaknesses.\
            Es necesario ser muy específico respecto a las strentghs y weaknesses, incorporando diferencias en skills requeridas vs que las que se muestran en el curriculum. \
            Entregar exactamente 5 strenghts y 5 weaknesses.'''
        
        instr_3='''como output entregar la información en formato JSON.'''
        
        instr_4='''A continuación te doy un ejemplo del output, no necesariamente deben ser las mismas frases: \
            data = {
                score: 75,  
                strength_1: "Sólida formación académica en áreas relevantes para el cargo.",
                strength_2: "Experiencia en liderazgo de equipos y proyectos exitosos.",
                strength_3: "Habilidades sólidas de comunicación y trabajo en equipo.",
                strength_4: "Capacidad para resolver problemas de manera creativa.",
                strength_5: "Adaptabilidad a entornos cambiantes y nuevas tecnologías."
                weakness_1: "Falta de experiencia en sql, python o R.",
                weakness_2: "No se observa experiencia en proyectos con equipos remotos, como lo solicita el job posting",
                weakness_3: "Falta de experiencia en aspectos de regulación laboral",
                weakness_4: "Sólo se ven cursos en finanzas corporativas, no experiencia laboral.",
                weakness_5: "No se ve experiencia en la industria retail"
                }
        '''
        
        #prompt with a chat model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #temperature=0.5,
            messages=[{"role": "system",
                       "content": role},
                      {"role": "user",
                       "content": instr_1 + instr_2 + "curriculum: " + text + "Cargo a postular:" + position_title + "Descripción cargo:" + description}]
        )
        
        response_content = response["choices"][0]["message"]["content"]
        st.write(response_content)
    with tab3:
        st.write("Hello world")
