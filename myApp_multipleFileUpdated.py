import streamlit as st
from streamlit_chat import message
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
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "DataChat", "content": "You are a helpful bot."}
    ]

if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []

if 'cost' not in st.session_state:
    st.session_state['cost'] = []

if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []

if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0


# sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation

st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")


# map model names to OpenAI model IDs

if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"


# reset everything

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


def save_uploadedfile(uploadedfile):
    with open(os.path.join("data/dataset", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return "data/dataset/" + uploadedfile.name


def generate_response(index, prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    response = index.query(prompt)
    st.session_state['messages'].append({"role": "DataChat", "content": response})

    last_token_usage = 0.0

    return response, last_token_usage


def get_loader(file_path_or_url):
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        handle_website = URLHandler()
        return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
    else:
        mime_type, _ = mimetypes.guess_type(file_path_or_url)

        if mime_type == 'application/pdf':
            return PyPDFLoader(file_path_or_url)
        elif mime_type == 'text/csv':
            return CSVLoader(file_path_or_url)
        elif mime_type in ['application/msword',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return UnstructuredWordDocumentLoader(file_path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")


def train_or_load_model(train, faiss_obj_path, file_path, idx_name):
    if train:
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        faiss_index = FAISS.from_documents(pages, embeddings)

        faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)


def answer_questions(faiss_index, user_input):
    messages = [
        SystemMessage(
            content='I want you to act as a document that I am having a conversation with. Your name is "AI '
                    'Assistant". You will provide me with answers from the given info. If the answer is not included, '
                    'say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about '
                    'the info. Never break character.')
    ]

    docs = faiss_index.similarity_search(query=user_input, k=2)

    main_content = user_input + "\n\n"
    for doc in docs:
        main_content += doc.page_content + "\n\n"

    messages.append(HumanMessage(content=main_content))
    ai_response = chat(messages).content
    messages.pop()
    messages.append(HumanMessage(content=user_input))
    messages.append(AIMessage(content=ai_response))

    return ai_response


df = None
uploaded_files = st.file_uploader("Choose files (PDF / CSV)", accept_multiple_files=True)
faiss_indices = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        uploaded_path = save_uploadedfile(uploaded_file)

        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
            agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        elif uploaded_file.type == "application/pdf":
            embeddings = OpenAIEmbeddings(openai_api_key=key)
            chat = ChatOpenAI(temperature=0, openai_api_key=key)
            faiss_obj_path = f"models/{uploaded_file.name}.pickle"
            index_name = uploaded_file.name
            faiss_index = train_or_load_model(1, faiss_obj_path, uploaded_path, index_name)
            faiss_indices[uploaded_file.name] = faiss_index
        else:
            st.write("Incompatible file type")


st.session_state['generated'] = []
st.session_state['past'] = []
st.session_state['messages'] = [
    {"role": "system", "content": "You are a helpful assistant."}
]
st.session_state['number_tokens'] = []
st.session_state['model_name'] = []
st.session_state['cost'] = []
st.session_state['total_cost'] = 0.0
st.session_state['total_tokens'] = []


# container for chat history

response_container = st.container()
# container for text box

container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        match = re.search(r'file: (\w+)', user_input)
        if match:
            file_name = match.group(1)
            faiss_index = faiss_indices.get(file_name)
            if faiss_index:
                output = answer_questions(faiss_index, user_input)

                total_tokens = 0
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['model_name'].append(model_name)
                st.session_state['total_tokens'].append(total_tokens)

                if model_name == "GPT-3.5":
                    cost = total_tokens * 0.002 / 1000
                else:
                    cost = total_tokens * 0.002 / 1000
                st.session_state['cost'].append(cost)
                st.session_state['total_cost'] += cost
            else:
                st.write(f"No file found with name '{file_name}'")
        else:
            st.write("Please specify a file name using the format 'file: FILE_NAME'")


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
