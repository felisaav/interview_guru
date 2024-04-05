import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
#import json

# functions
##Read pdf files
def read_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


st.title('AI interview Guru')
tab0, tab1, tab2,tab3,tab4 = st.tabs(["JobMatchRecruiter","ResumeRX", "JobMatchMaker","InterviewIQBoost", "Read Me"])

with tab0: #NLP group project
    with st.form(key ='Form_0'):
        uploaded_resume = st.file_uploader("Load resumes (in pdf format): ", type=['pdf'],accept_multiple_files=True)
        text=[]
        
        for uploaded_file in uploaded_resume:
            if uploaded_file is not None:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc:
                    text += page.get_text()
                #st.write(text) 
                doc.close()

        submit_code_0 = st.form_submit_button(label ="JobMatchRecruiter")
        
        if submit_code_0:
            st.write(text[0])

with tab1:
    with st.form(key ='Form_1'):
        uploaded_pdf = st.file_uploader("Load summary (in pdf format): ", type=['pdf'])
    
        if uploaded_pdf is not None:
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            #st.write(text) 
            doc.close()
        submit_code_1 = st.form_submit_button(label ="ResumeRX")

        if submit_code_1:
            # Load the key from a file
            api_key = st.secrets.key#open(st.secrets.key, 'r').read().strip('\n')
            assert api_key.startswith('sk-'), 'Error loding the API key. OpenAI API Keys start with "sk-".'
            openai.api_key = api_key

            #system role
            role="eres un reclutador de RRHH experto, que puede analizar en detalle el curriculum, \
                entregamdo insights para mejorar el currículum"
            
            #instructions
            instr_1_1="Entregar un score de 1 a 100 respecto a la calidad del curriculum."
            instr_1_2="Entregar 5 principales Fortalezas y 5 principales debilidades del curriculum, debe ser exacto el número de 5. En el caso de las debilidades entregar una propuesta de nueva redacción"
            instr_1_3="Entregar listado de las 5 principales skills técnicas a destacar y 5 skills no técnicas a destacar del currículum. Debe ser exacto el número 5"
            instr_1_4="Identificar idiona original del curriculum y responder en el mismo idioma. Si, por ejemplo, el currículum está en inglés las respuestas deben ser en inglés"
            #prompt with a chat model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0, #consistency in the answer
                messages=[{"role": "system",
                           "content": role},
                          {"role": "user",
                           "content": instr_1_1 + instr_1_2 + instr_1_3 + instr_1_4 + "curriculum: " + text }]
            )
            
            response_content_1 = response["choices"][0]["message"]["content"]
            st.write(response_content_1)


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
            temperature=0,
            messages=[{"role": "system",
                       "content": role},
                      {"role": "user",
                       "content": instr_1 + instr_2 + "curriculum: " + text + "Cargo a postular:" + position_title + "Descripción cargo:" + description}]
        )
        
        response_content = response["choices"][0]["message"]["content"]
        st.write(response_content)
    with tab3:
        st.write("""**Exciting updates in progress!** We're hard at work developing a new feature to enhance your interview preparation experience. While we're 
                    still working on it, stay tuned for an even more valuable resource to help you ace your interviews. We appreciate your patience and look
                    forward to bringing you the best interview preparation tools""")

