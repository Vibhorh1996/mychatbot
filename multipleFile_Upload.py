import os
import streamlit as st
import tiktoken
import faiss
import pypdf
import PyPDF2
from langdetect import detect
from iso639 import languages

# Create FAISS index
index = faiss.IndexFlatL2(768)

# Function to tokenize PDF content
def tokenize_pdf_content(uploaded_file):
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    num_pages = pdf_reader.numPages
    content = []

    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        text = page.extractText()
        content.append(text)

    return content

# Function to index PDF files
def index_pdf_files(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        tokens = tokenize_pdf_content(pdf_file)
        documents.append(tokens)
    doc_tensors = [faiss.tensor_from_array(doc, dim=1) for doc in documents]
    doc_ids = faiss.IndexIDMap(index)
    doc_ids.add_with_ids(faiss.cat(doc_tensors), np.arange(len(doc_tensors)))

# Function to search for query in the PDFs
def search_pdf(query):
    query_tokens = tiktoken.tokenize(query)
    query_tensor = faiss.tensor_from_array(query_tokens, dim=1)
    _, I = index.search(query_tensor, k=5)  # Search top 5 most similar documents
    results = []
    for i in I[0]:
        results.append(pdf_files[i])
    return results

# Streamlit app
st.title("PDF Search")
pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
if pdf_files:
    index_pdf_files(pdf_files)

query = st.text_input("Enter your query:")
if query:
    results = search_pdf(query)
    if results:
        st.write("Matching PDF files:")
        for result in results:
            st.write(result)
    else:
        st.write("No matching PDF files found.")
