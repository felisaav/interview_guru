import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
from openai import ChatCompletion
client = ChatCompletion()

import json

# functions
#-----------------------------------------------------------
#Preprocess job description and resumes with chat completion model
#Resumes
def preprocess_cv(cv):
    completion = client.create(
        model="gpt-3.5-turbo",
        temperature=0.2,#to receive more similar answers
        seed=123,
        messages=[
            {"role": "system", 
             "content": '''You are a helpful and expert HR recruiter assistant, with experience to analyze resumes.
             Provide your answer in JSON strcuture like this
             {"name":"name",
             "email":"email",
             "skills":[<list of skills>],
             "education":[<list of education>],
             "experience":"2 lines summary proffesional experience",
             "years_of_experience":"<Approximation of years of experience, it should be a range 0-3,4-7,8-10,10-15,+15>"}'''},
            {"role": "user", 
             "content":
                 f'''give me a list with the name, email, the exact 10 most relevant skills based on the entire resume, 
                 education, summary of professional experience, and
                 years of experience of the following text:{cv}'''},
        ]
    )
    return eval(completion.choices[0].message.content)

#Job descriptions
def preprocess_jd(jd):
    completion = client.create(
        model="gpt-3.5-turbo",
        temperature=0.2,#to receive more similar answers
        seed=123,
        messages=[
            {"role": "system", 
             "content": '''You are a helpful and expert HR recruiter assistant, with experience to analyze resumes.
             Provide your answer in JSON strcuture like this
             {"job title":"job title",
             "role":"summary of the role",
             "skills":[<list of skills that are required for this role>],
             "education":[<list of education requirements>],
             '''},
            {"role": "user", 
             "content":
                 f'''give me the title of the position, a summary of the role, the exact 10 most relevant skills based on the entire job description, and 
                 education requirements of:{jd}'''},
        ]
    )
    return completion.choices[0].message.content

# Embedding function and cosine metric
def embedding_vector(text):
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-3-small")
    
    embedding = response['data'][0]['embedding']
    return embedding

def similarity_score(embedding_1,embedding_2):
    a=np.dot(embedding_1, embedding_2)
    return(a)

def merge_columns(row):
    # Convert elements to strings before joining
    skills_str = ', '.join(map(str, row['skills']))
    education_str = ', '.join(map(str, row['education']))
    experience_str = str(row['experience'])  # Ensure experience is a string
    years_of_experience_str = str(row['years_of_experience'])  # Ensure years_of_experience is a string
    
    # Concatenate the strings with appropriate separators
    merged_info = skills_str + '; ' + education_str + '; ' + experience_str + '; ' + years_of_experience_str
    return merged_info
#-----------------------------------------------------------


st.title('AI interview Guru')
tab0, tab1, tab2,tab3,tab4 = st.tabs(["JobMatchRecruiter","ResumeRX", "JobMatchMaker","InterviewIQBoost", "Read Me"])

with tab0: #NLP group project
    with st.form(key ='Form_0'):
        uploaded_resume = st.file_uploader("Load resumes (in pdf format): ", type=['pdf'],accept_multiple_files=True)
        uploaded_job_description = st.file_uploader("Load Job Description (in pdf format): ", type=['pdf'])
        i=0
        text = []  # Initialize text as a list of lists
        
        for uploaded_file in uploaded_resume:
            if uploaded_file is not None:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text.append([])  # Append an empty list for each resume
                for page in doc:
                    text[i].append(page.get_text())  # Append text to the inner list
                doc.close()
                i += 1

        if uploaded_job_description is not None:
            doc = fitz.open(stream=uploaded_job_description.read(), filetype="pdf")
            job_description = ""
            for page in doc:
                job_description += page.get_text()
            doc.close()
            
        submit_code_0 = st.form_submit_button(label ="JobMatchRecruiter")
        
        if submit_code_0:
            api_key = st.secrets.key#open(st.secrets.key, 'r').read().strip('\n')
            assert api_key.startswith('sk-'), 'Error loding the API key. OpenAI API Keys start with "sk-".'
            openai.api_key = api_key
        
            #Process and embedding job description
            j_d=embedding_vector(job_description)

            #Analyze resumes
            #------------------------
            #resumes=pd.DataFrame()
            #for i in range(len(text)):
            #    if 'resumes' not in globals():
            #        resumes = pd.DataFrame(columns=["name", "email", "skills", "education", "experience", "years_of_experience"])
            #    # Append the new resume data to the DataFrame
            #    resumes = resumes.append(preprocess_cv(text[i]), ignore_index=True)

            #st.write(resumes)

            #resumes['merged_info'] = resumes.apply(merge_columns, axis=1)

            ## Apply the embedding_vector function to the "merged_info" column
            #resumes['embedding'] = resumes['merged_info'].apply(embedding_vector)
            
            ## Apply the similarity_score function between the resulting embeddings and the constant vector 'j_d'
            #resumes['score'] = resumes.apply(lambda row: similarity_score(row['embedding'], j_d), axis=1)
            
            #st.write(resumes.sort_values(by='score', ascending=False))
            #------------------------
            # Analyze resumes
            resumes = pd.DataFrame(columns=["name", "email", "skills", "education", "experience", "years_of_experience"])  # Initialize DataFrame

            print(type(resumes))  # Check the type of resumes
            print(resumes.head())  # Print the first few rows to inspect its structure

            #for i in range(len(text)):
            #    # Append the new resume data to the DataFrame
            #    new_row = preprocess_cv(text[i])
            #    resumes = resumes.append(new_row, ignore_index=True)
            
            ## Check the DataFrame structure
            #st.write(resumes)
            
            ## Further processing
            #resumes['merged_info'] = resumes.apply(merge_columns, axis=1)
            #resumes['embedding'] = resumes['merged_info'].apply(embedding_vector)
            #resumes['score'] = resumes.apply(lambda row: similarity_score(row['embedding'], j_d), axis=1)
            
            ## Display the sorted DataFrame
            #st.write(resumes.sort_values(by='score', ascending=False))





            

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

