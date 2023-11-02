import streamlit as st
import pandas as pd
import fitz


def extract_data(uploaded_file):
    # Initialize a variable to store the extracted text
    pymupdf_text = ""

    # Open the PDF file
    pdf_document = fitz.open(path)

    # Iterate through each page and extract text
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text()
        pymupdf_text += text

    # Close the PDF document
    pdf_document.close()

    return pymupdf_text
    
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    df = extract_data(uploaded_file)
    st.write(df)


