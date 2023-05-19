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
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)


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
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []


# load chat model

def load_chat_model():
    agent = create_pandas_dataframe_agent()
    gpt_model = OpenAIEmbeddings()
    faiss_model = FAISS()
    document_loaders = {
        "CSV": CSVLoader,
        "PDF": PyPDFLoader(file_path_or_url),
        "Web Page": WebBaseLoader,
    }
    document_loaders = {
        key: document_loaders[key]()
        for key in document_loaders
    }
    return ChatOpenAI(
        agent=agent,
        gpt_model=gpt_model,
        faiss_model=faiss_model,
        document_loaders=document_loaders,
    )


# process user input and generate response

def generate_response(user_input):
    if user_input:
        user_input = user_input.strip()
        st.session_state['messages'].append(HumanMessage(content=user_input))

        document_type = st.session_state['document_type']
        if document_type:
            document_loader = st.session_state['document_loaders'][document_type]

            if document_type == "Web Page":
                websites = document_loader.websites
                links = URLHandler.extract_links_from_websites(websites)
                document_loader.websites = links

            document_loader.refresh_documents()
            messages = st.session_state['messages']
            response = chat_model.process_message_batch(messages)

            for message in response:
                if isinstance(message, AIMessage):
                    st.session_state['generated'].append(message.content)
                elif isinstance(message, SystemMessage):
                    st.write(f"[System]: {message.content}")

            st.session_state['messages'] = []
            st.session_state['generated'] = []


# render user interface

st.sidebar.header('Configuration')

document_type = st.sidebar.selectbox('Document Type', ['CSV', 'PDF', 'Web Page'])
st.session_state['document_type'] = document_type

if document_type:
    document_loader = st.session_state['document_loaders'].get(document_type)
    if document_loader is None:
        document_loader = document_loaders[document_type]()
        st.session_state['document_loaders'][document_type] = document_loader

    if document_type == "CSV":
        csv_files = st.sidebar.file_uploader('Upload CSV', type='csv', accept_multiple_files=True)
        if csv_files:
            document_loader.csv_files = csv_files
            generate_response('')

    elif document_type == "PDF":
        pdf_files = st.sidebar.file_uploader('Upload PDF', type='pdf', accept_multiple_files=True)
        if pdf_files:
            document_loader.file_path_or_url = pdf_files[0]
            generate_response('')

    elif document_type == "Web Page":
        websites = st.sidebar.text_area("Enter Website URLs (one per line)")
        document_loader.websites = websites.strip().split('\n')
        generate_response('')


st.subheader('Chat')

user_input = st.text_input('You:', '')

if st.button('Query'):
    generate_response(user_input)

for generated in st.session_state['generated']:
    st.write(f"[AI]: {generated}")
