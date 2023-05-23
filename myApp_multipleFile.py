import streamlit as st
import os
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlsplit
import mimetypes
import pickle
import tiktoken
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


class DocumentLoader:
    def __init__(self, file_path_or_url):
        self.file_path_or_url = file_path_or_url

    def load_and_split(self):
        if self.file_path_or_url.startswith("http://") or self.file_path_or_url.startswith("https://"):
            handle_website = URLHandler()
            return WebBaseLoader(handle_website.extract_links_from_websites([self.file_path_or_url]))
        else:
            mime_type, _ = mimetypes.guess_type(self.file_path_or_url)

            if mime_type == 'application/pdf':
                return PyPDFLoader(self.file_path_or_url)
            elif mime_type == 'text/csv':
                return CSVLoader(self.file_path_or_url)
            elif mime_type in ['application/msword',
                               'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return UnstructuredWordDocumentLoader(self.file_path_or_url)
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")


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


# Setting page title and header
st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

# Set API Key
key = st.text_input('OpenAI API Key', '', type='password')
os.environ['OPENAPI_API_KEY'] = key
os.environ['OPENAI_API_KEY'] = key

# Model selection
st.sidebar.markdown('### Select Model')
model_name = st.sidebar.radio('Model', ('GPT-3.5', 'GPT-4'))
model_id = 'gpt-3.5-turbo' if model_name == 'GPT-3.5' else 'gpt-4.0-turbo'

# Initialize components
if not 'openai_agent' in st.session_state:
    st.session_state.openai_agent = None

if not 'faiss_index' in st.session_state:
    st.session_state.faiss_index = None

if not 'embedding' in st.session_state:
    st.session_state.embedding = None

if not 'document_loaders' in st.session_state:
    st.session_state.document_loaders = []

if not 'total_tokens' in st.session_state:
    st.session_state.total_tokens = 0

if not 'cost' in st.session_state:
    st.session_state.cost = 0.0

# File upload
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)

if uploaded_files:
    document_loaders = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        file_path = f"uploaded_file{file_extension}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        document_loader = DocumentLoader(file_path)
        document_loaders.append(document_loader)

    st.session_state.document_loaders = document_loaders
    st.session_state.document_models = []
    st.session_state.total_tokens = 0
    st.session_state.cost = 0.0

# Check if API Key is provided
if not key:
    st.warning("Please provide your OpenAI API Key.")
    st.stop()

# Check if any document loaders are available
if not st.session_state.document_loaders:
    st.info("Please upload files to begin.")
    st.stop()

# Load the documents
if not st.session_state.document_models:
    document_models = []
    for document_loader in st.session_state.document_loaders:
        document_model = document_loader.load_and_split()
        document_models.append(document_model)
    st.session_state.document_models = document_models

# Load the Faiss Index and Embedding
if not st.session_state.faiss_index or not st.session_state.embedding:
    faiss_index = FAISS.load("faiss.index")
    st.session_state.faiss_index = faiss_index

    embedding = OpenAIEmbeddings("gpt-3.5-turbo")
    st.session_state.embedding = embedding

# Initialize the chat agent
if not st.session_state.openai_agent:
    model = ChatOpenAI(model_id=model_id)
    st.session_state.openai_agent = model

# Initialize OpenAI
openai_agent = st.session_state.openai_agent

# Initialize Faiss Index and Embedding
faiss_index = st.session_state.faiss_index
embedding = st.session_state.embedding

# Total tokens and cost calculation
total_tokens = st.session_state.total_tokens
cost_per_token = 0.0004  # Cost per token for GPT-3.5 Turbo, adjust as per your usage
cost = st.session_state.cost

# Chat interface
user_input = st.text_area("You:", "", height=100, max_chars=200, key="input_text_area")
if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        user_message = HumanMessage(user_input)
        for document_model in st.session_state.document_models:
            document_model.receive(user_message)

        document_responses = []
        for document_model in st.session_state.document_models:
            document_response = document_model.generate(
                [AIMessage(agent=embedding, score_threshold=0.5)], beam_size=1
            )
            document_responses.append(document_response.generated_messages[0].content)

        document_answer = "\n".join(document_responses)
        total_tokens += count_tokens(user_input)
        total_tokens += count_tokens(document_answer)
        cost = total_tokens * cost_per_token

        response = openai_agent.respond(user_input)
        response_message = response.generated_messages[0]
        response_content = response_message.content

        for document_model in st.session_state.document_models:
            document_model.receive(AIMessage(agent=openai_agent, score_threshold=0.5))
            document_model.receive(AIMessage(agent=embedding, score_threshold=0.5))

        st.write("Response:", response_content)
        st.write("Answer:", document_answer)
        st.write(f"Total Tokens: {total_tokens}")
        st.write(f"Cost: ${cost:.2f}")
        st.session_state.total_tokens = total_tokens
        st.session_state.cost = cost

# Save the state
st.session_state.faiss_index = faiss_index
st.session_state.embedding = embedding
st.session_state.openai_agent = openai_agent
st.session_state.document_loaders = st.session_state.document_loaders
st.session_state.document_models = st.session_state.document_models
st.session_state.total_tokens = total_tokens
st.session_state.cost = cost
