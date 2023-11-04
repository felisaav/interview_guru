import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
import json
#import plotly.graph_objects as go
#import matplotlib.pyplot as plt

st.title('AI interview Guru')
with st.form(key ='Form1'):
    uploaded_pdf = st.file_uploader("Load summary (in pdf format): ", type=['pdf'])

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
        No entregar exactamente 5 strenghts y 5 weaknesses.'''
    
    instr_3='''como output entregar la información con el siguiente formato JSON.'''
    
    instr_4='''A continuación te doy un ejemplo: \
        data = {
            "score": 75,  
            "strengths": {
                "strength_1": "Sólida formación académica en áreas relevantes para el cargo.",
                "strength_2": "Experiencia en liderazgo de equipos y proyectos exitosos.",
                "strength_3": "Habilidades sólidas de comunicación y trabajo en equipo.",
                "strength_4": "Capacidad para resolver problemas de manera creativa.",
                "strength_5": "Adaptabilidad a entornos cambiantes y nuevas tecnologías."
            },
            "weaknesses": {
                "weakness_1": "Falta de experiencia en sql, python o R.",
                "weakness_2": "No se observa experiencia en proyectos con equipos remotos, como lo solicita el job posting",
                "weakness_3": "Falta de experiencia en aspectos de regulación laboral",
                "weakness_4": "Necesidad de mejorar la gestión del tiempo y la planificación.",
                "weakness_5": "Enfrentar dificultades para hablar en público y presentar ideas de manera efectiva."
            }
        }
    '''
    
    #prompt with a chat model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        #max_tokens= 500,
        messages=[{"role": "system",
                   "content": role},
                  {"role": "user",
                   "content": instr_1 + instr_2 + instr_3 + "curriculum:"+text + "Cargo a postular:"+ position_title + "Descripción cargo:"+description}]
    )
    
    response_content = response["choices"][0]["message"]["content"]
    #data = json.loads(response_content)
    st.write(response_content)
    #----------------------------
    # Create a list of dictionaries in the desired format
    #concept_detail_value = []

    # Add the "score" as the first entry
    #concept_detail_value.append({"Concept": "Score", "Detail": "Score", "Value": data["score"]})
    
    # Add the strengths and weaknesses
    #for category, category_data in data.items():
    #    if category in ["strengths", "weaknesses"]:
    #        for key, value in category_data.items():
    #            concept_detail_value.append({"Concept": category.capitalize(), "Detail": key, "Value": value})
    
    # Create the DataFrame
    #df = pd.DataFrame(concept_detail_value)
    #df["Value"] = df["Value"].astype(str)
    
    # Display Results
    #st.markdown('-------')
    #st.subheader('Result of the analysis')
    #score = df[df['Concept'] == 'Score']['Value'].values[0]
    #st.write('Your march score is: ' + score + '/100')
    #st.markdown('-------')
    #st.write('**Main Strenghts**')
    # Filter the DataFrame to get the rows with "Concept" starting with "Strength_"
    #strengths_df = df[df['Concept'].str.match(r'Strength(_\d+)?')]
    # Now, you have the DataFrame containing strengths
    # You can display them in Streamlit
    #for index, row in strengths_df.iterrows():
    #    st.write(f"{row['Detail']}: {row['Value']}")
    #st.markdown('-------')
    #st.write('**Main Improvements to do**')
    # Filter the DataFrame to get the rows with "Concept" starting with "Weakness_"
    #weakness_df = df[df['Concept'].str.match(r'Weakness(_\d+)?')]
    # Now, you have the DataFrame containing weakness
    # You can display them in Streamlit
    #for index, row in strengths_df.iterrows():
    #    st.write(f"{row['Detail']}: {row['Value']}")

