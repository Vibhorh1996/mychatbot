import streamlit as st
from streamlit_chat import message
import pandas as pd
# from llama_index.indices.struct_store import GPTPandasIndex
# from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex
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
os.environ['OPENAI_API_KEY'] = key


# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "DataChat", "content": "You are a helpful bot."}
    ]

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

if 'current_link' not in st.session_state:
    st.session_state['current_link'] = None

if 'current_loader' not in st.session_state:
    st.session_state['current_loader'] = None


def process_system_message():
    if len(st.session_state['messages']) > 0:
        last_message = st.session_state['messages'][-1]
        if 'type' in last_message:
            if last_message['type'] == 'system':
                system_message = last_message['content']
                if 'extract' in system_message:
                    return system_message['extract']

    return None


def process_human_message(human_message):
    if len(st.session_state['messages']) > 0:
        last_message = st.session_state['messages'][-1]
        if 'type' in last_message:
            if last_message['type'] == 'human':
                human_message = last_message['content']
                if 'action' in human_message:
                    return human_message['action']

    return None


def process_model_output(model_output):
    system_message = {
        "role": "DataChat",
        "type": "system",
        "content": model_output,
    }

    st.session_state['messages'].append(system_message)


def process_loader_message(loader_message):
    if 'type' in loader_message:
        if loader_message['type'] == 'system':
            if 'content' in loader_message:
                if 'loading_links' in loader_message['content']:
                    st.info("Loading links...")


def display_link_input():
    st.subheader("Link Input")
    link_input = st.text_input("Enter a website URL")
    if link_input:
        if st.button("Load Links"):
            st.session_state['current_link'] = link_input
            st.session_state['current_loader'] = 'url'


def display_text_input():
    st.subheader("Text Input")
    text_input = st.text_area("Enter your text")
    if text_input:
        if st.button("Process Text"):
            st.session_state['current_link'] = text_input
            st.session_state['current_loader'] = 'text'


def display_file_input():
    st.subheader("File Input")
    file_input = st.file_uploader("Upload a file")
    if file_input:
        if st.button("Process File"):
            file_contents = file_input.read()
            file_extension = os.path.splitext(file_input.name)[1].lower()

            if file_extension == ".pdf":
                st.session_state['current_loader'] = 'pdf'
                st.session_state['current_link'] = file_contents
            elif file_extension == ".csv":
                st.session_state['current_loader'] = 'csv'
                st.session_state['current_link'] = file_contents
            elif file_extension == ".docx":
                st.session_state['current_loader'] = 'word'
                st.session_state['current_link'] = file_contents
            else:
                st.error("Unsupported file format.")


def display_link_preview():
    if st.session_state['current_link']:
        st.subheader("Link Preview")
        st.write(st.session_state['current_link'])


def process_message(input_message):
    with message(messages=st.session_state['messages'], add_message=input_message):
        if input_message['role'] == 'User':
            if input_message['content'] == '/restart':
                st.session_state['messages'] = [
                    {"role": "DataChat", "content": "You are a helpful bot."}
                ]
                st.session_state['generated'] = []
                return

            if input_message['content'] == '/extract_links':
                process_loader_message(
                    {
                        "type": "system",
                        "content": {
                            "loading_links": True
                        },
                    }
                )

                links = URLHandler.extract_links_from_websites(
                    [st.session_state['current_link']]
                )

                message_data = [
                    {"role": "DataChat", "content": f"Extracted {len(links)} links."}
                ]

                for link in links:
                    message_data.append({"role": "DataChat", "content": link})

                st.session_state['messages'].extend(message_data)
                return

            if input_message['content'] == '/load_links':
                process_loader_message(
                    {
                        "type": "system",
                        "content": {
                            "loading_links": True
                        },
                    }
                )

                links = URLHandler.extract_links_from_websites(
                    [st.session_state['current_link']]
                )

                if st.session_state['current_loader'] == 'pdf':
                    st.session_state['current_loader'] = 'url_pdf'
                elif st.session_state['current_loader'] == 'csv':
                    st.session_state['current_loader'] = 'url_csv'
                elif st.session_state['current_loader'] == 'word':
                    st.session_state['current_loader'] = 'url_word'

                st.session_state['current_link'] = links

                message_data = [
                    {"role": "DataChat", "content": f"Loaded {len(links)} links."}
                ]

                for link in links:
                    message_data.append({"role": "DataChat", "content": link})

                st.session_state['messages'].extend(message_data)
                return

            if st.session_state['current_loader'] == 'url':
                st.session_state['current_loader'] = 'url_text'

            if st.session_state['current_loader'] == 'url_pdf':
                st.session_state['current_loader'] = 'pdf'

            if st.session_state['current_loader'] == 'url_csv':
                st.session_state['current_loader'] = 'csv'

            if st.session_state['current_loader'] == 'url_word':
                st.session_state['current_loader'] = 'word'

            if st.session_state['current_loader'] == 'url_text':
                st.session_state['current_loader'] = 'text'

            if st.session_state['current_loader'] == 'text':
                data_loader = create_pandas_dataframe_agent(
                    st.session_state['current_link']
                )
            elif st.session_state['current_loader'] == 'pdf':
                data_loader = PyPDFLoader(st.session_state['current_link'])
            elif st.session_state['current_loader'] == 'csv':
                data_loader = CSVLoader(st.session_state['current_link'])
            elif st.session_state['current_loader'] == 'word':
                data_loader = UnstructuredWordDocumentLoader(
                    st.session_state['current_link']
                )

            document = data_loader.load_and_split()

            loader_message = {
                "role": "DataChat",
                "type": "system",
                "content": {"document_loaded": True},
            }

            st.session_state['messages'].append(loader_message)

            st.session_state['past'] = []

            message_data = [
                {"role": "DataChat", "content": f"Loaded {len(document)} rows."}
            ]

            for row in document:
                message_data.append({"role": "DataChat", "content": row})

            st.session_state['messages'].extend(message_data)

            return

    tokens = count_tokens(st.session_state['messages'], model="gpt-3.5-turbo")

    if tokens >= 4096:
        message_data = [
            {
                "role": "DataChat",
                "type": "system",
                "content": "The conversation has reached the maximum token limit (4096 tokens). Please restart the conversation.",
            }
        ]
        st.session_state['messages'].extend(message_data)
        return

    if input_message['role'] == 'DataChat':
        input_message['role'] = 'System'

    st.session_state['messages'].append(input_message)

    if input_message['role'] == 'User':
        system_message = {
            "role": "System",
            "type": "system",
            "content": {"input_message": input_message},
        }

        st.session_state['messages'].append(system_message)


def display_messages():
    for message in st.session_state['messages']:
        if message['role'] == 'User':
            st.text_input("User", value=message['content'], disabled=True)
        elif message['role'] == 'DataChat':
            st.text_input("DataChat", value=message['content'], disabled=True)
        elif message['role'] == 'System':
            if 'type' in message and message['type'] == 'system':
                st.info(message['content'])
            else:
                st.text_input("System", value=message['content'], disabled=True)


def main():
    if st.session_state['model_name']:
        st.markdown(f"<h3 stype='text-align:center;'>{st.session_state['model_name']}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 stype='text-align:center;'>GPT-3.5 Turbo</h3>", unsafe_allow_html=True)

    st.sidebar.header("Options")
    option = st.sidebar.selectbox(
        "Choose an option",
        ("Link Input", "Text Input", "File Input", "Link Preview", "Chat"),
    )

    if option == "Link Input":
        display_link_input()
    elif option == "Text Input":
        display_text_input()
    elif option == "File Input":
        display_file_input()
    elif option == "Link Preview":
        display_link_preview()
    elif option == "Chat":
        display_messages()

        user_input = st.text_input("User", value="", key="user_input")
        if st.button("Send"):
            user_message = {"role": "User", "content": user_input}
            process_message(user_message)

    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = ""

    st.sidebar.header("Model Selection")
    model_option = st.sidebar.selectbox(
        "Choose a model",
        ("gpt-3.5-turbo", "gpt-4.0-turbo"),
    )

    if model_option != st.session_state['model_name']:
        st.session_state['model_name'] = model_option
        st.session_state['messages'] = [
            {"role": "DataChat", "content": "You are a helpful bot."}
        ]


if __name__ == "__main__":
    main()
