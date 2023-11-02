import streamlit as st
import pandas as pd

import fitz  # PyMuPDF

path = "Resume_FelipeSaavedra_2022.pdf"

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

# Now you can use the pymupdf_text variable as needed
print(pymupdf_text)  # Optionally, you can print the extracted text
