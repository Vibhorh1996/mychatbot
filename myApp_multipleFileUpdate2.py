import streamlit as st
import pandas as pd
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
import re


def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    return tokens


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


# setting page title and header
st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

# set API Key
key = st.text_input('OpenAI API Key', '', type='password')
os.environ['OPENAPI_API_KEY'] = key

# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = {}

if 'past' not in st.session_state:
    st.session_state['past'] = {}

if 'messages' not in st.session_state:
    st.session_state['messages'] = {
        "DataChat": "You are a helpful bot."
    }

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = {}

if 'cost' not in st.session_state:
    st.session_state['cost'] = {}

if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0

if 'selected_file' not in st.session_state:
    st.session_state['selected_file'] = None

if 'document_index' not in st.session_state:
    st.session_state['document_index'] = None

# document loaders
document_loaders = {
    "PDF": PyPDFLoader(),
    "CSV": CSVLoader(),
    "Web Page": WebBaseLoader(),
}

# select document type
document_type = st.sidebar.selectbox("Select Document Type", list(document_loaders.keys()))

# document upload
uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

# load and index documents
if uploaded_files:
    document_loader = document_loaders[document_type]

    if document_type == "Web Page":
        urls = [f.name for f in uploaded_files]
        links = URLHandler.extract_links_from_websites(urls)
        for link in links:
            document_loader.add_document(link)
    else:
        for uploaded_file in uploaded_files:
            content = uploaded_file.read()
            document_loader.add_document(content)

    # process and index the documents
    document_loader.process_documents()
    index = document_loader.index_documents()

    # store the index for later use
    st.session_state['document_index'] = index

# query input
query = st.text_input("Enter your query")

# query execution
if st.button("Query"):
    index = st.session_state['document_index']

    if index is not None:
        result = index.query(query)

        st.markdown("<h3>Results:</h3>", unsafe_allow_html=True)
        for doc_id, score in result:
            st.write(f"**Document ID**: {doc_id}, **Score**: {score}")

            # get the document content based on the document type
            if document_type == "Web Page":
                content = document_loader.get_document_by_url(doc_id)
            else:
                content = document_loader.get_document_by_id(doc_id)

            st.write(f"**Content**: {content}")
    else:
        st.write("Please upload documents first.")


