import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
#import requests as rq
from streamlit_lottie import st_lottie
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
    # Calculate the magnitudes of the embedding vectors
    norm_1 = np.linalg.norm(embedding_1)
    norm_2 = np.linalg.norm(embedding_2)
    # Calculate the cosine similarity score
    cosine_similarity = a / (norm_1 * norm_2)
    return cosine_similarity

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
# Title section

left_column, right_column = st.columns(2)
with left_column:
    st.title('AI interview Guru')
with right_column:
    #with st.echo():
    st_lottie("https://lottie.host/0d33f259-f4ca-4fca-bbdf-6d9796af6cf5/iG5NDmi7WE.json", height=300)
    #lottie_coding=load_lottieurl("https://lottie.host/0d33f259-f4ca-4fca-bbdf-6d9796af6cf5/iG5NDmi7WE.json")
#with right_column:
#	st_lottie(lottie_coding, height=300, key="coding")
    


#st.title('AI interview Guru')
#-----------------------------------------------------------

tab0, tab1 = st.tabs(["JobMatchRecruiter", "Read Me"])

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
            resumes=pd.DataFrame()
            for i in range(len(text)):
                # Create a DataFrame to store the resumes if it doesn't exist yet
                if 'resumes' not in globals():
                    resumes = pd.DataFrame(columns=["name", "email", "skills", "education", "experience", "years_of_experience"])
                # Append the new resume data to the DataFrame
                resumes = resumes.append(preprocess_cv(text[i]), ignore_index=True)
            
            # Apply the function to each row to create the new column
            resumes['merged_info'] = resumes.apply(merge_columns, axis=1)

            # Apply the embedding_vector function to the "merged_info" column
            resumes['embedding'] = resumes['merged_info'].apply(embedding_vector)
            
            # Apply the similarity_score function between the resulting embeddings and the constant vector 'j_d'
            resumes['score'] = resumes.apply(lambda row: similarity_score(row['embedding'], j_d), axis=1)
            
            # Display selected columns of the DataFrame sorted by 'score'
            st.title("Resume Scoring")
            st.write(resumes[['score', 'name', 'email', 'skills', 'experience', 'years_of_experience']].sort_values(by='score', ascending=False))

    with tab1:
        st.write("Welcome to AI Interview Guru, your ultimate tool for streamlining the job application process and enhancing interview preparation using cutting-edge artificial intelligence technologies.")

        st.header("Main Models")
        st.subheader("ChatCompletion Model")
        st.write("The ChatCompletion model, powered by OpenAI's GPT-3.5, is used throughout AI Interview Guru to assist HR recruiters and job seekers in **Preprocessing job descriptions and resumes**, **parsing resumes to extract relevant insights**")
        
        st.subheader("Embedding Vector Model")
        st.write("This model, text-embedding-3-small, is tailored for **generating embeddings**, which are **numerical representations of textual data**. Embeddings capture the semantic meaning of words, phrases, or entire documents in a high-dimensional vector space.")

        st.subheader("Scoring Explanation")
        st.markdown("Cosine Similarity Score")
        st.write("The cosine similarity score measures the cosine of the angle between two vectors in a multidimensional space. In the context of AI Interview Guru, it quantifies the similarity between the embeddings of a resume and a job description. The formula for cosine similarity is as follows:")

        st.latex(r'''
                \frac{\left|A\right|*\left|B\right|}{A*B}
                ''')
        
        st.write('''
                Where:
                - A and B are the embedding vectors of the resume and job description, respectively.
                - |A| and |B| represent the magnitudes (or norms) of the vectors.
                ''')
        
        st.subheader("Usage")
        st.write("JobMatchRecruiter: Upload resumes and job descriptions to match candidates with suitable positions. Receive detailed scoring and insights to make informed hiring decisions.")

