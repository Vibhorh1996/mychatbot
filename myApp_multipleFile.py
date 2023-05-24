# import necessary packages
import os
import re
import json
import PyPDF2
import streamlit as st
from pathlib import Path
import faiss
import pickle
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, urljoin
import requests
import mimetypes
from abc import ABC, abstractmethod
from typing import List
from src.parse_document import PdfParser
from src.indexer import FaissIndexer
from src.openai_embeddings import OpenAIEmbeddings, ChatOpenAI

"""
This is a Streamlit-based application that works as a chatbot for conversing with data from PDF files.
"""

# setting page title and header
#st.set_page_config(page_title="Data Chat", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A Chatbot for conversing with your data</h2>", unsafe_allow_html=True)

# set API Key
key = st.text_input('OpenAI API Key','',type='password')
os.environ['OPENAPI_API_KEY'] = key

# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
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
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

# class definitions

class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass


# class FAISS(BaseFAISS):
#     def save(self, file_path):
#         with open(file_path, "wb") as f:
#             pickle.dump(self, f)

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


# file uploading and processing

def save_uploadedfile(uploadedfile):
     with open(os.path.join("data/dataset", uploadedfile.name), "wb") as f:
         f.write(uploadedfile.getbuffer())
     return "data/dataset/" + uploadedfile.name


def generate_response(index, prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    response = index.query(prompt)
    st.session_state['messages'].append({"role": "DataChat", "content": response})

    # last_token_usage = index.llm_predictor.last_token_usage
    last_token_usage = 0.0
    # print(f"last_token_usage={last_token_usage}")

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


def train_or_load_model(train, faiss_obj_path, file_paths):
    if train:
        pages = [] # a list to store all the pages from all the files
        for file_path in file_paths: # loop through the file paths
            loader = get_loader(file_path) # get the loader for each file
            pages.extend(loader.load_and_split()) # load and split the pages and append them to the list

        faiss_index = FAISS.from_documents(pages, embeddings) # create a single FAISS index from all the pages
        faiss_index.save(faiss_obj_path) # save the FAISS index to a specific location

        return FAISS.load(faiss_obj_path) # return the loaded FAISS index
    else:
        return FAISS.load(faiss_obj_path) # return the loaded FAISS index


def answer_questions(faiss_index, user_input):
    messages = [
        SystemMessage(
            content='I want you to act as a document that I am having a conversation with. Your name is "AI '
                    'Assistant". You will provide me with answers from the given info. If the answer is not included, '
                    'say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about '
                    'the info. Never break character.')
    ]

    # while True:
        # question = input("Ask a question (type 'stop' to end): ")
        # if question.lower() == "stop":
        #     break

    docs = faiss_index.similarity_search(query=user_input, k=2) # search for the most similar documents based on the user input

    main_content = user_input + "\n\n" # create a main content string with the user input
    for doc in docs: # loop through the matched documents
        main_content += doc.page_content + "\n\n" # append the page content to the main content string

    messages.append(HumanMessage(content=main_content)) # append a human message with the main content to the messages list
    ai_response = chat(messages).content # generate an AI response using the chat model and the messages list
    messages.pop() # remove the last message from the list
    messages.append(HumanMessage(content=user_input)) # append a human message with the user input to the list
    messages.append(AIMessage(content=ai_response)) # append an AI message with the AI response to the list

    return ai_response # return the AI response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

# allow users to upload multiple PDF files using the st.file_uploader function
uploaded_files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files: # if there are uploaded files
    # initialize OpenAI embeddings and chat model
    embeddings = OpenAIEmbeddings(openai_api_key=key, model_name = model)
    chat = ChatOpenAI(temperature=0, openai_api_key=key, model_name = model)

    faiss_obj_path = "models/all_files.pickle" # define the path to save or load the FAISS object for all files
    file_paths = [] # a list to store all the file paths of uploaded files
    for uploaded_file in uploaded_files: # loop through the uploaded files
        file_path = save_uploadedfile(uploaded_file) # save each file to a specific location and get its path
        file_paths.append(file_path) # append each file path to the list

    faiss_index = train_or_load_model(1, faiss_obj_path, file_paths) # train or load a single FAISS index for all files

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100) # create a text area for user input
        submit_button = st.form_submit_button(label='Send') # create a button for submitting user input

    if submit_button and user_input: # if user input is submitted
        output = answer_questions(faiss_index, user_input)  # use the answer_questions function to generate a response based on user input and FAISS index   
        total_tokens = 0 # set total tokens to zero (this can be changed if token usage is available)
        st.session_state['past'].append(user_input) # append user input to session state variable 'past'
        st.session_state['generated'].append(output) # append generated output to session state variable 'generated'
        st.session_state['model_name'].append(model_name)  # append model name to session state variable 'model_name'
        st.session_state['total_tokens'].append(total_tokens)  # append total tokens to session state variable 'total_tokens'

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = total_tokens * 0.002 / 1000
        st.session_state['cost'].append(cost)  # append cost to session state variable 'cost'
        st.session_state['total_cost'] += cost  # update total cost in session state variable 'total_cost'

if st.session_state['generated']:  # if there are generated responses
    with response_container:  # in response container
        for i in range(len(st.session_state['generated'])):  # loop through each response
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')  # display user message using message function
            message(st.session_state["generated"][i], key=str(i))  # display DataChat message using message function
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")  # display model name, token count and cost information
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")  # update total cost in sidebar

# function to display messages in a chat-like format
def message(text, is_user=False, key=None):
    if is_user:
        color = "#008080"
    else:
        color = "#000000"

    html_text = f"""
    <div style="display:flex; flex-direction:row; margin-bottom:10px;">
      <div style="width:50px; height:50px; border-radius:25px; background-color:{color}; margin-right:10px;"></div>
      <div style="display:flex; flex-direction:column;">
        <div style="font-size:16px; font-weight:bold; color:{color}; margin-bottom:5px;">{'You' if is_user else 'DataChat'}</div>
        <div style="font-size:14px; color:#000000;">{text}</div>
      </div>
    </div>
    """
    return st.markdown(html_text, unsafe_allow_html=True, key=key)

# main function to run the app (not needed since everything is already done in global scope)
# def main():
#     pass

# if __name__ == "__main__":
#     main()
