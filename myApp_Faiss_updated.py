# Import streamlit and other libraries
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

# Import a library for summarizing text
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor # You can also use text-summarizer or any other library

# Import a library for extracting text from PDF files
from pdfminer.high_level import extract_text # You can also use pdfminer2 or pdfminer.six

# Define a function to count tokens using tiktoken library
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    return tokens

# Define an abstract class for document loaders using ABC module
class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass

# Define a subclass of FAISS that can save and load itself using pickle module
class FAISS(BaseFAISS):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

# Define a class for handling URLs using requests and BeautifulSoup modules
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

# Set page title and header using streamlit markdown 
st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

# Set API Key using streamlit text input 
key = st.text_input('OpenAI API Key', '', type='password')
os.environ['OPENAPI_API_KEY'] = key

# Initialize session state variables using streamlit session state 
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

if 'faiss_indices' not in st.session_state:
    st.session_state['faiss_indices'] = {}

if 'agents' not in st.session_state:
    st.session_state['agents'] = {}

# Add a session state variable to store all the uploaded files and their contents in a dictionary 
if 'uploaded_files_dict' not in st.session_state:
    st.session_state['uploaded_files_dict'] = {}

# Define a function to save the uploaded file using os module 
def save_uploadedfile(uploadedfile):
    with open(os.path.join("data/dataset", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return "data/dataset/" + uploadedfile.name

# Define a function to summarize a PDF file using text-summarizer library 
def summarize_pdf(file_path):
    # Import text-summarizer library 
    from text_summarizer import Summarizer

    # Open the PDF file and extract the text using pdfminer.six library 
    from pdfminer.high_level import extract_text
    text = extract_text(file_path)

    # Print the text variable to check if it contains the text from the PDF file 
    print(text)

    # Create an instance of the summarizer object using text-summarizer library 
    #summary = summarizer.Summarizer()
    summary = Summarizer()

    # Call the summarize method of the summarizer object with the extracted text as input and get the summary as output 
    result = summary.summarize(text)

    # Print the result variable to check if it contains a summary of the text 
    print(result)

    # Return the result as output 
    return result

# Define a function to get the loader object based on the file path or url using mimetypes and langchain.document_loaders modules 
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

# Define a function to train or load a model based on the train flag using langchain.embeddings and langchain.vectorstores modules 
def train_or_load_model(train, faiss_obj_path, file_path, idx_name):
    if train:
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        faiss_index = FAISS.from_documents(pages, embeddings)

        faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)

# Define a function to answer questions using FAISS index and OpenAI chat model 
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

# Create an instance of OpenAI embeddings and chat model using langchain.embeddings and langchain.chat_models modules 
embeddings = OpenAIEmbeddings(openai_api_key=key)
chat = ChatOpenAI(temperature=0, openai_api_key=key)

# Let user upload files using streamlit file uploader 
uploaded_files = st.file_uploader("Choose files (PDF / CSV)", accept_multiple_files=True)
if uploaded_files:
     for uploaded_file in uploaded_files:
         file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
         uploaded_path = save_uploadedfile(uploaded_file)

         if uploaded_file.type == "text/csv":
             df = pd.read_csv(uploaded_file)
             st.dataframe(df.head(10))
             agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
             st.session_state['agents'][uploaded_file.name] = agent
         elif uploaded_file.type == "application/pdf":
             faiss_obj_path = f"models/{uploaded_file.name}.pickle"
             index_name = uploaded_file.name
             # name = uploaded_file.name
             faiss_index = train_or_load_model(1, faiss_obj_path, uploaded_path, index_name)
             st.session_state['faiss_indices'][uploaded_file.name] = faiss_index
 
             # Add the uploaded file and its content to the dictionary 
             st.session_state['uploaded_files_dict'][uploaded_file.name] = extract_text(uploaded_path)


# Define a variable to store the summaries of all the uploaded files 
summaries = []

# Loop through the uploaded files and check if they are PDF files. If yes, call the summarize_pdf function with the file path as input and get the summary as output. Append the summary to the summaries list. 
for uploaded_file in uploaded_files:
    if uploaded_file.type == "application/pdf":
         # Get the file path of the uploaded_file using save_uploadedfile function
        file_path = save_uploadedfile(uploaded_file)
        # Call the summarize_pdf function with the file_path as input
        summary = summarize_pdf(file_path)
        summaries.append(summary)

# Let user choose a model from GPT-3.5 or GPT-4 using streamlit sidebar radio 
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))

# Map model names to OpenAI model IDs 
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# Show the total cost of the current conversation using streamlit sidebar write 
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

# Let user clear the current conversation using streamlit sidebar button 
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Reset everything if clear button is clicked 
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
 
    # Reset the uploaded files dictionary as well 
    st.session_state['uploaded_files_dict'] = {}

# Create a container for chat history using streamlit container 
response_container = st.container()

# Create a container for text box using streamlit container 
container = st.container()

# Let user enter their input and send it using streamlit form and text area 
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        # Process the user input and generate a response if submit button is clicked and user input is not empty 
        if submit_button and user_input:
            if uploaded_files:
                # Loop through the uploaded files dictionary and check if the user input matches any file name. If yes, use the corresponding FAISS index or agent object to answer the question. 
                for file_name, file_content in st.session_state['uploaded_files_dict'].items():
                    if user_input.lower() == file_name.lower():
                        if file_name.endswith(".csv"):
                            agent = st.session_state['agents'][file_name]
                            output = agent.run(user_input)
                        elif file_name.endswith(".pdf"):
                            faiss_index = st.session_state['faiss_indices'][file_name]
                            output = answer_questions(faiss_index, user_input)
                        else:
                            output = "Sorry, I don't know how to handle this file type."
                        break
 
                # If no file name matches, use the FAISS index of all the uploaded files to find the most similar documents to the user input and use them to answer the question. 
                else:
                    faiss_index_all = FAISS.from_documents(list(st.session_state['uploaded_files_dict'].values()), embeddings)
                    output = answer_questions(faiss_index_all, user_input)

                # Count the tokens and calculate the cost of the output using tiktoken and count_tokens functions 
                total_tokens = count_tokens(output, model=model)
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

                # Display the output and the cost using streamlit message and write functions 
                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated"][i], key=str(i))
                            st.write(
                                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
