import streamlit as st
import pandas as pd
import openai
import os
import fitz  # PyMuPDF
import numpy as np
import requests as rq
from streamlit_lottie import st_lottie
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import re
import spacy
from spacy.matcher import Matcher
from dateutil.parser import parse

data = {
    "Name": [],
    "Email": [],
    "Phone number": [],
    "Skills": [],
    "Education": []
}

skills_job = []

###
# FUNCTION SECTION
###
def load_lottieurl(url):
    r = rq.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document,
                         sentences_count=3)  # You can adjust the number of sentences in the summary
    return ' '.join(str(sentence) for sentence in summary)


###
def process_summary_files(uploaded_files):
    data = {
        "Name": [],
        "Email": [],
        "Phone number": [],
        "Skills": [],
        "Education": []
    }

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            text = ""
            if uploaded_file.type == 'text/plain':
                doc = fitz.open(stream=uploaded_file.read(), filetype="txt")
                for page in doc:
                    text += page.get_text()
                doc.close()

            elif uploaded_file.type == 'application/pdf':
                try:
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in doc:
                        text += page.get_text("text")  # Use "text" for plain text extraction
                    doc.close()

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
            else:
                st.warning(f"Unsupported file format: {uploaded_file.type}")

            # Procesar el texto aquí
            keywords = extract_keywords(text)
            name = extract_name(text)
            phone_number = extract_mobile_number(text)
            email = extract_email(text)
            skills_set = extract_skills(text)
            education = extract_education(text)

            # Agregar los datos extraídos al diccionario
            data["Name"].append(name)
            data["Email"].append(email)
            data["Phone number"].append(phone_number)
            data["Skills"].append(", ".join(skills_set))
            data["Education"].append(", ".join([str(edu) for edu in education]))

    return data


def process_job_files(uploaded_job_files):
    job_skills = []

    if uploaded_job_files is not None:
        for file in uploaded_job_files:
            text = ""
            if file.type == 'text/plain':
                doc = fitz.open(stream=file.read(), filetype="txt")
                for page in doc:
                    text += page.get_text()
                doc.close()
            elif file.type == 'application/pdf':
                try:
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    for page in doc:
                        text += page.get_text()  # Use "text" for plain text extraction
                    doc.close()
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
            else:
                st.warning(f"Unsupported file format: {uploaded_file.type}")

            # Aquí puedes agregar el procesamiento para otros tipos de archivo si es necesario

            skills_job = extract_skills(text)
            job_skills.extend(skills_job)

    return job_skills


def calculate_score(skills, desired_skills):
    total_skills = len(desired_skills)
    matched_skills = sum(skill in desired_skills for skill in skills)

    if total_skills == 0:
        return 0
    else:
        score = (matched_skills / total_skills) * 100
        rounded_score = round(score, 2)  # Redondear el puntaje a dos decimales
        return rounded_score

# Create two columns
left_column, right_column = st.columns(2)

# Now you can use left_column and right_column
with left_column:
    st.title('AI interview Guru')

with right_column:
    lottie_coding=load_lottieurl("https://lottie.host/0d33f259-f4ca-4fca-bbdf-6d9796af6cf5/iG5NDmi7WE.json")
with right_column:
	st_lottie(lottie_coding, height=300, key="coding")

#Menu Display
tab1, tab2, CV_Job_Comparassion, Cover_letter_gen, tab3,tab4, tab5, tab6 = \
    st.tabs(["NLP-CV Parser", "Job Analysis", "CV_Job_Comparassion", "Cover_letter_gen",
            "ResumeRX",
            "JobMatchMaker",
            "InterviewIQBoost",
            "Read Me"])

with tab1:
    def extract_keywords(text):
        # Tokenize the text into words
        words = word_tokenize(text)

        # Filter out stopwords
        stop_words = set(stopwords.words("english"))
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

        # Calculate frequency distribution
        freq_dist = nltk.FreqDist(filtered_words)

        # Get the 5 most common keywords
        keywords = freq_dist.most_common(5)

        return [keyword[0] for keyword in keywords]


    # load pre-trained model
    nlp = spacy.load('en_core_web_sm')

    # initialize matcher with a vocab
    matcher = Matcher(nlp.vocab)


    def extract_name(resume_text):
        nlp_text = nlp(resume_text)

        # First name and Last name are always Proper Nouns
        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

        matcher.add('NAME', [pattern])

        matches = matcher(nlp_text)

        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text


    def extract_mobile_number(text):
        phone = re.findall(re.compile(
            r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
            text)

        if phone:
            number = ''.join(phone[0])
            if len(number) > 10:
                return '+' + number
            else:
                return number


    def extract_email(email):
        email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None


    # load pre-trained model
    nlp = spacy.load('en_core_web_sm')


    def extract_skills(resume_text):
        print("extract skills----------------")
        nlp_text = nlp(resume_text)

        # removing stop words and implementing word tokenization
        tokens = [token.text.lower() for token in nlp_text if not token.is_stop]

        # reading the csv file
        skills = pd.read_csv("skills.csv")

        # extract values
        skills = [skill.lower() for skill in list(skills.columns.values)]  # Lowercase skills for comparison
        skills = [skill.strip() for skill in skills]  # Remove whitespace from skills

        print(skills)
        skillset = []

        # check for one-grams (example: python) using case-insensitive matching
        for token in tokens:
            if token in skills:
                print("Found skill:", token)
                skillset.append(token)

        # check for bi-grams and tri-grams (example: machine learning)
        for chunk in nlp_text.noun_chunks:
            token = chunk.text.lower().strip()
            if token in skills:
                skillset.append(token)

        return [i.capitalize() for i in set(skillset)]  # Capitalize unique skills


    ## EDUCATION

    # load pre-trained model
    nlp = spacy.load('en_core_web_sm')

    # Grad all general stop words
    STOPWORDS = set(stopwords.words('english'))

    # Education Degrees
    EDUCATION = [
        'BE', 'B.E.', 'B.E', 'BS', 'B.S',
        'ME', 'M.E', 'M.E.', 'MS', 'M.S',
        'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
        'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
    ]


    def extract_education(resume_text):
        nlp_text = nlp(resume_text)

        # Sentence Tokenizer
        nlp_text = [sent.text.strip() for sent in nlp_text.sents]

        edu = {}
        # Extract education degree
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                # Replace all special symbols
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.upper() in EDUCATION and tex not in STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1] if index + 1 < len(nlp_text) else text

        # Extract year
        education = []
        for key in edu.keys():
            year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
            if year:
                education.append((key, ''.join(year[0])))
            else:
                education.append(key)
        return education



    with st.form(key='Form_CV_Summary'):
        uploaded_files = st.file_uploader("Load summary (in txt or pdf format): ", type=['txt', 'pdf'],
                                          accept_multiple_files=True)
        CV_Summary_Code = st.form_submit_button(label="CV_Summary")
        if CV_Summary_Code:

            if uploaded_files is not None:
                text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == 'text/plain':
                        doc = fitz.open(stream=uploaded_file.read(), filetype="txt")
                        for page in doc:
                            text += page.get_text()
                        doc.close()

                    elif uploaded_file.type == 'application/pdf':
                        try:
                            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                            for page in doc:
                                text += page.get_text("text")  # Use "text" for plain text extraction
                            doc.close()

                        except Exception as e:
                            st.error(f"Error processing PDF: {e}")
                    else:
                        st.warning(f"Unsupported file format: {uploaded_file.type}")

                    # Process the text here
                    keywords = extract_keywords(text)
                    name = extract_name(text)
                    phone_number = extract_mobile_number(text)
                    email = extract_email(text)
                    skills_set = extract_skills(text)
                    education = extract_education(text)


                    # Append the extracted data to the dictionary
                    data["Name"].append(name)
                    data["Email"].append(email)
                    data["Phone number"].append(phone_number)
                    data["Skills"].append(", ".join(skills_set))
                    data["Education"].append(", ".join([str(edu) for edu in education]))

        st.header("CV Summary")
        st.table(pd.DataFrame(data))


with tab2:
    with st.form(key='Form_6'):
        uploaded_files = st.file_uploader("Load job (in txt format): ", type=['txt'],
                                          accept_multiple_files=True)

        text = ""

        JOB_button = st.form_submit_button(label="Analyize Job Description")
        if JOB_button:
            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == 'text/plain':
                        doc = fitz.open(stream=uploaded_file.read(), filetype="txt")
                        for page in doc:
                            text += page.get_text()
                        doc.close()


            skills_2 = extract_skills(text)
            if skills_2:
                st.write("Job Skills Found:", ", ".join(skills_2))
            else:
                st.write("Job Skills not found")

with CV_Job_Comparassion: #CV_Job_Comparassion
    with st.form(key='Form_CV_Job_Comparassion'):
        uploaded_files = st.file_uploader("Load summary (in txt or pdf format): ", type=['txt', 'pdf'],
                                          accept_multiple_files=True)
        uploaded_job_files = st.file_uploader("Load JOB (in txt or pdf format): ", type=['txt', 'pdf'],
                                              accept_multiple_files=True)
        CV_Summary_Code = st.form_submit_button(label="CV Job Comparassion")
        if CV_Summary_Code:
            data = process_summary_files(uploaded_files)
            job_skills = process_job_files(uploaded_job_files)

            scores = []
            for skills in data["Skills"]:
                score = calculate_score(skills.split(", "), job_skills)
                scores.append(score)

            data["Score"] = scores

            st.header("CV Analysis and Job Match")
            st.write("Table with candidates CV compare with the job description propose with a column score indicateing how this candidate fit")
            df = pd.DataFrame(data)
            df['Score'] = df['Score'].round(2)
            df_sorted = df.sort_values(by='Score', ascending=False)
            st.table(df_sorted)
            st.write("Job Skills Found:", ", ".join(job_skills) if job_skills else "Job Skills not found")
with Cover_letter_gen:
    with st.form(key ='Form_Cover_letter_gen'):
        uploaded_files = st.file_uploader("Load summary (in txt or pdf format): ", type=['txt', 'pdf'],
                                          accept_multiple_files=True)
        uploaded_job_files = st.file_uploader("Load JOB (in txt or pdf format): ", type=['txt', 'pdf'],
                                              accept_multiple_files=True)
        CV_Summary_Code = st.form_submit_button(label="Cover letter Generator")
        if CV_Summary_Code:
            st.header("Cover letter generate")
            # Use a pipeline as a high-level helper
            with open('key.txt', 'r') as file:
                api_key = file.read().strip()

            assert api_key.startswith('sk-'), 'Error loding the API key. OpenAI API Keys start with "sk-".'
            openai.api_key = api_key

            # system role
            role = "eres un reclutador de RRHH experto, que puede analizar en detalle el curriculum, \
                entregamdo insights para mejorar el currículum"

            # instructions
            instr_1_1 = "Entregar un score de 1 a 100 respecto a la calidad del curriculum."
            instr_1_2 = "Entregar 5 principales Fortalezas y 5 principales debilidades del curriculum, debe ser exacto el número de 5. En el caso de las debilidades entregar una propuesta de nueva redacción"
            instr_1_3 = "Entregar listado de las 5 principales skills técnicas a destacar y 5 skills no técnicas a destacar del currículum. Debe ser exacto el número 5"
            instr_1_4 = "Identificar idiona original del curriculum y responder en el mismo idioma. Si, por ejemplo, el currículum está en inglés las respuestas deben ser en inglés"
            # prompt with a chat model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,  # consistency in the answer
                messages=[{"role": "system",
                           "content": role},
                          {"role": "user",
                           "content": instr_1_1 + instr_1_2 + instr_1_3 + instr_1_4 + "curriculum: " + text}]
            )

            response_content_1 = response["choices"][0]["message"]["content"]
            st.write(generated_text)
with tab3: #ResumeRX
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
            api_key = st.secrets.key  # open(st.secrets.key, 'r').read().strip('\n')
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


with tab4: #JobMatchMaker
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

    with tab5:
        st.write("""**Exciting updates in progress!** We're hard at work developing a new feature to enhance your interview preparation experience. While we're 
                    still working on it, stay tuned for an even more valuable resource to help you ace your interviews. We appreciate your patience and look
                    forward to bringing you the best interview preparation tools""")

    with tab6:
            st.write("""
            # AI Interview Guru

            AI Interview Guru is a comprehensive suite of tools designed to empower recruiting journey. 
            AI Interview Guru helps recruiting shine at every stage: compare in a table all candidates, review a CV, search for best job matching. 

            ## About

            This tool is developed by: 
            * **Felipe Saavedra**                    
            * **Julián Gaona**  
            * **Víctor Vargas** 
            * **Marina Dufour** 
             
            
            We're working hard to bring you the best interview preparation tools. Stay tuned for more updates!

            ## Contribution

            - Contributions from other team members are also valuable and appreciated.
            - Feel free to contribute to this project by submitting pull requests or sharing your ideas.

            We appreciate your support and patience.

            ## Contact

            For any inquiries or feedback, please reach out to us at nlp@student.ie.edu

            Thank you for your interest in our project!
            """)

            st.subheader("Key Features")

            with st.expander("CV Parser and Summary"):
                st.write("""
                - Upload your CV in text or PDF format.
                - Get an automatic summary of your key skills and experiences, saving you time and effort.
                """)

            with st.expander("Resume RX"):
                st.write("""
                - Analyze your existing resume for effectiveness.
                - Receive tailored recommendations to optimize your resume for specific roles.
                - Craft a compelling resume that stands out to hiring managers.
                """)

            with st.expander("Job Matching Tool"):
                st.write("""
                - Provide your skills and desired job criteria.
                - Discover suitable job openings based on advanced AI matching.
                - Save valuable time by focusing on the most relevant opportunities.
                """)

            st.subheader("Benefits")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Save Time and Effort:**")
                st.write("Streamline your interview preparation process with automated features.")

            with col2:
                st.write("**Gain Insights:**")
                st.write("Receive objective feedback on your resume to improve its effectiveness.")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Increase Confidence:**")
                st.write("Approach job applications with a strong and optimized resume.")

            with col2:
                st.write("**Land the Job:**")
                st.write("Find the most relevant job openings aligned with your skillset and aspirations.")

            st.subheader("How to Use")

            st.write("""
            Launch AI Interview Guru: Access the application (instructions on how to access will depend on deployment method).

            Navigate Features: Utilize the intuitive interface to explore individual features (CV Parser, Resume RX, Job Matching Tool).

            Provide Input: Upload your CV or enter relevant information as required by each tool.

            Generate Results: Receive automated summaries, recommendations, or job listings based on your input.
            """)









                



