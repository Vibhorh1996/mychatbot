import streamlit as st
from streamlit_chat import message
import pandas as pd
from streamlit.uploaded_file_manager import UploadedFile
import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import List
from langchain.agents import create_pandas_dataframe_agent
import requests
import mimetypes
from bs4 import BeautifulSoup
import tiktoken
from urllib.parse import urljoin, urlsplit
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as BaseFAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

def save_uploadedfiles(uploaded_files: List[UploadedFile]) -> List[str]:
    file_paths = []
    for uploaded_file in uploaded_files:
        with open(os.path.join("data/dataset", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append("data/dataset/" + uploaded_file.name)
    return file_paths

class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass

class FAISS(BaseFAISS):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed_url = urlsplit(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)

    @staticmethod
    def extract_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                if URLHandler.is_valid_url(absolute_url) and (
                        absolute_url.startswith("http://") or absolute_url.startswith("https://")):
                    links.append(absolute_url)

        return links

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = []

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.extend(links)

        return all_links

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    return tokens

def train_or_load_model(train, faiss_obj_path, file_paths, idx_name):
    if train:
        loaders = []
        for file_path in file_paths:
            if file_path.startswith("http://") or file_path.startswith("https://"):
                handle_website = URLHandler()
                loaders.append(WebBaseLoader(handle_website.extract_links_from_websites([file_path])))
            else:
                mime_type, _ = mimetypes.guess_type(file_path)

                if mime_type == 'application/pdf':
                    loaders.append(PyPDFLoader(file_path))
                elif mime_type == 'text/csv':
                    loaders.append(CSVLoader(file_path))
                elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    loaders.append(UnstructuredWordDocumentLoader(file_path))
                else:
                    loaders.append(PyPDFLoader(file_path))

        df = create_pandas_dataframe_agent(loaders)
        faiss_obj = FAISS(nlist=2, method="Flat", space="cosine")
        faiss_obj.train(df, "docs", "sentence", idx_name)
        faiss_obj.save(faiss_obj_path)
    else:
        faiss_obj = FAISS.load(faiss_obj_path)

    return faiss_obj

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_faiss_obj(train, faiss_obj_path, file_paths, idx_name):
    faiss_obj = train_or_load_model(train, faiss_obj_path, file_paths, idx_name)
    return faiss_obj

def get_similar_documents(user_input, faiss_obj, idx_name):
    retrieval_results = faiss_obj.retrieval(user_input, "docs", "sentence", idx_name)
    return retrieval_results

st.title("Document Similarity Search")

train = st.checkbox("Train Model")
uploaded_files = st.file_uploader("Choose files (PDF / CSV)", accept_multiple_files=True)
file_paths = None

if uploaded_files:
    file_details = [{"FileName": uploaded_file.name, "FileType": uploaded_file.type} for uploaded_file in uploaded_files]
    file_paths = save_uploadedfiles(uploaded_files)

    for file_detail in file_details:
        st.write(file_detail)

    # Process the files and perform necessary operations

faiss_obj_path = "data/faiss_objects/faiss_obj.pkl"
idx_name = "index"

faiss_obj = get_faiss_obj(train, faiss_obj_path, file_paths, idx_name)

user_input = st.text_input("Enter your query")

if user_input:
    retrieval_results = get_similar_documents(user_input, faiss_obj, idx_name)
    st.write(retrieval_results)
