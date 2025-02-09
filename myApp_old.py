# import the required libraries
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
import time
from bs4 import BeautifulSoup
import tiktoken
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urljoin, urlsplit
from langchain.agents import create_pandas_dataframe_agent
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

# initialize the session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if "number_tokens" not in st.session_state:
    st.session_state["number_tokens"] = []
if "model_name" not in st.session_state:
    st.session_state["model_name"] = []
if "cost" not in st.session_state:
    st.session_state["cost"] = []
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = []

# set up the user interface
st.set_page_config(page_title="Data Chat", page_icon=":robot_face:")
st.markdown("<h1 style='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown(
    "<h2 style='text-align:center;'>A Chatbot for conversing with your data</h2>",
    unsafe_allow_html=True,
)

# set API Key
key = st.text_input('OpenAI API Key','',type='password')
os.environ['OPENAPI_API_KEY'] = key
os.environ['OPENAI_API_KEY'] = key

# create a sidebar
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(
    f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
)
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything if the clear button is pressed
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state["number_tokens"] = []
    st.session_state["model_name"] = []
    st.session_state["cost"] = []
    st.session_state["total_cost"] = 0.0
    st.session_state["total_tokens"] = []
    counter_placeholder.write(
        f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
    )

# define a function to save each uploaded file with a unique name
def save_uploadedfile(uploadedfile):
    # add a timestamp or a random string to the file name
    file_name = uploadedfile.name + "_" + str(int(time.time()))
    with open(os.path.join("data/dataset", file_name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return "data/dataset/" + file_name

# create a file uploader that accepts multiple files of only PDF type
uploaded_files = st.file_uploader(
    "Choose a file (PDF)", type="pdf", accept_multiple_files=True
)
pdf_files = [uploaded_file for uploaded_file in uploaded_files]

# define a class for loading and splitting documents from different sources
class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass

# define a subclass of BaseFAISS that implements methods for saving and loading the FAISS index
class FAISS(BaseFAISS):
    __module__ = __name__
    __qualname__ = "FAISS"

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

# define a function to get the appropriate loader for the file path based on the mime type
def get_loader(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == "application/pdf":
        return PyPDFLoader(file_path)
    elif mime_type == "text/csv":
        return CSVLoader(file_path)
    elif mime_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        return UnstructuredWordDocumentLoader(file_path)
    elif mime_type is None:
        # try to infer the format from the file content
        with open(file_path, "rb") as f:
            data = f.read()
            if data.startswith(b"%PDF"):
                return PyPDFLoader(file_path)
            elif data.startswith(b"PK"):
                return UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")

# define a function to train or load the FAISS model based on a flag and save it to a file path
def train_or_load_model(train, faiss_obj_path, file_path, idx_name):
    if train:
        loader = get_loader(file_path)
        pages = loader.load_and_split()

        faiss_index = FAISS.from_documents(pages, embeddings)

        faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)

# Create a TF-IDF vectorizer object
vectorizer = TfidfVectorizer()

def extract_text_from_pdfs(pdf_files):
    # Initialize an empty list to store the text content of each PDF file
    text_content = []

    # Iterate over the list of PDF files
    for pdf_file in pdf_files:
        # Create a PyPDFLoader object with the file path
        loader = PyPDFLoader(pdf_file)

        # Load and split the PDF file into a list of documents
        documents = [save_uploadedfile(document) for document in loader.load_and_split()]

        # Initialize an empty string to store the text content of the PDF file
        pdf_text = ""

        # Iterate over the documents
        for document in documents:
            # Extract the page content from the document and append it to the pdf_text string
            pdf_text += document.page_content

        # Append the pdf_text string to the text_content list
        text_content.append(pdf_text)

    # Return the text_content list
    return text_content


def vectorize(text, pdf_files):
    # Extract the text content from the PDF files using our custom function
    pdf_content = extract_text_from_pdfs(pdf_files)

    # Fit the vectorizer on the PDF content
    vectorizer.fit(pdf_content)

    # Transform the text into a TF-IDF vector
    vector = vectorizer.transform([text])

    # Return the vector
    return vector


def calculate_similarity(user_input, ai_response, document_content):
    # Convert the user input and the AI response into vectors
    user_vector = vectorize(user_input, document_content)  # pass the document content as the second argument
    ai_vector = vectorize(ai_response, document_content)  # pass the document content as the second argument

    # Calculate the cosine similarity between the vectors
    similarity = 1 - cosine(user_vector, ai_vector)

    # Return the similarity score
    return similarity


def answer_questions(file_paths, user_input):
    best_score = 0.0
    best_response = ""

    for file_path in file_paths:
        # Create a PyPDFLoader object with the file path
        loader = PyPDFLoader(file_path)

        # Load and split the PDF file into a list of documents
        documents = loader.load_and_split()

        # Iterate over the documents
        for document in documents:
            # Extract the page content from the document
            page_text = document.page_content

            # Generate an AI response for the user's query using the page text
            messages = [
                SystemMessage(
                    content='You are a document named "AI Assistant". Answer from the info or say "Hmm, I am not sure." No other questions.'
                ),
                HumanMessage(content=user_input),
                AIMessage(content=page_text)  # pass the string value
            ]

            ai_response = chat(messages).content

            # Update the best score and response if a higher similarity is found
            similarity_score = calculate_similarity(user_input, ai_response, document)  # pass the document as the second argument

            if similarity_score > best_score:
                best_score = similarity_score
                best_response = ai_response

    return best_response

# create an OpenAI object
openai = OpenAI()

# create an OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

# create a ChatOpenAI object
chat = ChatOpenAI()

# create a list of file paths and a list of faiss_index objects
file_paths = []
faiss_indices = []

# save each uploaded file and create a faiss_index object for each file
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        uploaded_path = save_uploadedfile(uploaded_file)
        file_paths.append(uploaded_path)
        faiss_obj_path = uploaded_path + ".faiss"
        idx_name = uploaded_path + ".idx"
        train = 1
        faiss_index = train_or_load_model(train, faiss_obj_path, uploaded_path, idx_name)
        faiss_indices.append(faiss_index)

# create a container for the chat history and another container for the text box
response_container = st.container()
container = st.container()

# create a form with a text area and a submit button to get the user input
with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

# generate a response from the index or the agent based on the user input and append it to the session state messages
if submit_button and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
      # Call the modified answer_questions function
    best_response = answer_questions(file_paths, user_input)

    # Display the file name or path of the document that contains the answer
    st.write(f"Answer from: {file_paths[best_index]}")

    # Append the answer to the session state messages
    st.session_state["messages"].append({"role": "DataChat", "content": best_response})

 # iterate over the faiss_index objects and perform a similarity search with the user input for each one
#     scores = []
#     responses = []
#     for faiss_index in faiss_indices:
#         docs = faiss_index.similarity_search(query=user_input, k=2)
#         # append the scores and responses of the documents to the lists
#         scores.append(docs[0][1])  # Access the similarity score of the first document
#         responses.append(docs[0][0].page_content)


#     # compare the scores and select the best one as the answer
#     best_score = max(scores)
#     best_index = scores.index(best_score)
#     best_response = responses[best_index]


#     # display the file name or path of the document that contains the answer
#     st.write(f"Answer from: {file_paths[best_index]}")

#    # append the answer to the session state messages
#     st.session_state["messages"].append({"role": "DataChat", "content": best_response})

   # generate an AI response from the messages using the chat model
    ai_response = chat(st.session_state["messages"]).content

   # append the user input and the AI response to the session state messages
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "AI Assistant", "content": ai_response})

   # append the user input and the AI response to the session state past and generated lists
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(ai_response)

   # get the last token usage from the chat model
    last_token_usage = chat.last_token_usage

   # append the model name and the last token usage to the session state model_name and total_tokens lists
    st.session_state["model_name"].append(model_name)
    st.session_state["total_tokens"].append(last_token_usage)

   # calculate the cost of the response based on the model name and the last token usage
    if model_name == "GPT-3.5":
      cost = last_token_usage * 0.00000006
    else:
      cost = last_token_usage * 0.00000012

   # append the cost to the session state cost list
    st.session_state["cost"].append(cost)

   # update the total cost in the session state
    st.session_state["total_cost"] += cost

   # display the chat history and the text box using streamlit widgets
    if st.session_state["generated"]:
      with response_container:
          for i in range(len(st.session_state["generated"])):
              message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
              message(st.session_state["generated"][i], key=str(i))
              st.write(
                  f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}"
            )
              counter_placeholder.write(
                  f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
            )
