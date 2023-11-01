import streamlit as st
import pandas as pd
import PyPDF2

# Define a function to extract data from a PDF file
def extract_data(uploaded_file):
    # Create a PDF file reader
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)

    # Initialize an empty list to store extracted data
    data = []

    # Iterate through each page in the PDF
    for page_num in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_num)
        text = page.extract_text()
        data.append(text)

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=["Extracted Text"])
    return df

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    df = extract_data(uploaded_file)
    st.dataframe(df)


