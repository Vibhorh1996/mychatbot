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


class FAISS:
    def __init__(self):
        self.indexes = {}

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.indexes, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            obj = FAISS()
            obj.indexes = pickle.load(f)
            return obj

    def add_index(self, document_name, index):
        self.indexes[document_name] = index

    def get_index(self, document_name):
        return self.indexes.get(document_name)


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
key = st.text_input('OpenAI API Key','',type='password')
os.environ['OPENAPI_API_KEY'] = key
os.environ['OPENAI_API_KEY'] = key


# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated']=[]

if 'past' not in st.session_state:
    st.session_state['past']=[]

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {"role":"DataChat","content":"You are a helpful bot."}
    ]

if 'model_name' not in st.session_state:
    st.session_state['model_name']=[]

if 'cost' not in st.session_state:
    st.session_state['cost']=[]

if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens']=[]

if 'total_cost' not in st.session_state:
    st.session_state['total_cost']=0.0

# sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:",("GPT-3.5", "GPT-4"))
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
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
        
# def askQuestion():
#     prompt = st.text_input("write your question:")
#     response = index.query(prompt)
#     st.write(response)

#     # Get the last token usage
#     last_token_usage = index.llm_predictor.last_token_usage
#     st.write(f"last_token_usage={last_token_usage}")

def save_uploadedfiles(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        with open(os.path.join("data/dataset", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append("data/dataset/" + uploaded_file.name)
        st.write(file_paths)
    return file_paths


def generate_response(index,prompt):
    st.session_state['messages'].append({"role":"user","content":prompt})

    response = index.query(prompt)
    st.session_state['messages'].append({"role":"DataChat","content":response})
    st.write(response)

    #last_token_usage = index.llm_predictor.last_token_usage
    last_token_usage = 0.0
    #print(f"last_token_usage={last_token_usage}")

    return response,  last_token_usage

def get_loader(file_path_or_url):
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        handle_website = URLHandler()
        return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
    else:
        mime_type, _ = mimetypes.guess_type(file_path_or_url)

        if mime_type == 'application/pdf':
            st.write(mime_type)
            return PyPDFLoader(file_path_or_url)
        elif mime_type == 'text/csv':
            return CSVLoader(file_path_or_url)
        elif mime_type in ['application/msword',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return UnstructuredWordDocumentLoader(file_path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

# def get_loader(file_path_or_url):
#     if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
#         handle_website = URLHandler()
#         return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
#     else:
#         mime_type, _ = mimetypes.guess_type(file_path_or_url)

#         if mime_type == 'application/pdf':
#             return PyPDFLoader(file_path_or_url)
#         elif mime_type == 'text/csv':
#             return CSVLoader(file_path_or_url)
#         elif mime_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
#             return UnstructuredWordDocumentLoader(file_path_or_url)
#         else:
#             raise ValueError(f"Unsupported file type: {mime_type or 'unknown'}")

def train_or_load_model(train, faiss_obj_path, file_paths, idx_name):
    if train:
        loaders = [get_loader(file_path) for file_path in file_paths]
        pages = []
        for loader in loaders:
            pages.extend(loader.load_and_split())

        faiss_index = FAISS()
        for i, page in enumerate(pages):
            # Perform any necessary processing or feature extraction on the page

            # Add the processed page to the FAISS index
            faiss_index.add_index(f"document_{i}", page)

        faiss_index.save(faiss_obj_path)

        return faiss_index
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


# def main():
#     faiss_obj_path = "/Users/puneetsachdeva/Downloads/langchain-chat-main/models/test.pickle"
#     file_path = "/Users/puneetsachdeva/Downloads/langchain-chat-main/data/mlb_players.csv"
#     index_name = "test"

#     train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
#     faiss_index = train_or_load_model(train, faiss_obj_path, file_path, index_name)
#     answer_questions(faiss_index)


df=None
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)

if uploaded_files is not None:
    file_paths = save_uploadedfiles(uploaded_files)
    st.write(file_paths)
    
    for file_path in file_paths:
        # Perform processing for each file
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        file_extension = file_path.split(".")[-1].lower()
        st.write(file_extension)
        
        if file_extension == "csv":
            df = pd.read_csv(file_path)
            st.dataframe(df.head(10))
            agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        elif file_extension == "pdf":
            embeddings = OpenAIEmbeddings(openai_api_key=key)
            st.write(embeddings)
            chat = ChatOpenAI(temperature=0, openai_api_key=key)
            st.write(chat)
            # train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
            faiss_obj_path = "models/test.pickle"
            index_name = "test"
            faiss_index = train_or_load_model(1, faiss_obj_path, file_path, index_name)
            st.write(faiss_index)
            # answer_questions(faiss_index)
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

# documents = SimpleDirectoryReader('data/dataset').load_data()
# index = GPTSimpleVectorIndex.from_documents(documents)

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # output, last_token_count = generate_response(index,user_input)
        output = None
        if file_extension == "text/csv":
            st.write(output)
        elif file_extension == "application/pdf":
            st.write(output)
            output = answer_questions(faiss_index, user_input)    
        #st.write(output)
        #total_tokens = last_token_count
        total_tokens = 0
        st.session_state['past'].append(user_input)
        # st.session_state['generated'].append(output.response)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            #cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            cost = total_tokens * 0.002 / 1000
        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    #st.write(st.session_state['generated'])
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
