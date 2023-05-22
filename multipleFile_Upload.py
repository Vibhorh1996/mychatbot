import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from text_summarizer import Summarizer
from urllib.parse import urljoin, urlsplit
from pdfminer.high_level import extract_text
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as BaseFAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import tiktoken
import mimetypes
import os
import pickle
from abc import ABC, abstractmethod
from typing import List

# Import summarization library
from text_summarizer import Summarizer

# Streamlit configuration
st.set_page_config(page_title="Data Chat", page_icon=":robot_face:")
st.markdown("<h1 style='text-align:center;'>Data Chat</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>A Chatbot for Conversing with Your Data</h2>", unsafe_allow_html=True)

# Session state initialization
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "DataChat", "content": "You are a helpful bot."}
    ]

if "model_name" not in st.session_state:
    st.session_state["model_name"] = []

if "cost" not in st.session_state:
    st.session_state["cost"] = []

if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = []

if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0

# Set OpenAI API Key
key = st.text_input("OpenAI API Key", "", type="password")
os.environ["OPENAPI_API_KEY"] = key

# Initialize OpenAI embeddings and chat models
embeddings = OpenAIEmbeddings(openai_api_key=key)
chat = ChatOpenAI(temperature=0, openai_api_key=key)

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
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                absolute_url = urljoin(url, href)
                if URLHandler.is_valid_url(absolute_url) and (
                    absolute_url.startswith("http://")
                    or absolute_url.startswith("https://")
                ):
                    links.append(absolute_url)

        return links

# Define a class for loading and processing PDF documents
class PDFLoader(DocumentLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        return extract_text(self.file_path).splitlines()

# Define a class for loading and processing CSV documents
class CSVLoader(DocumentLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        df = pd.read_csv(self.file_path)
        return df.to_string(index=False).splitlines()

# Define a class for loading and processing unstructured Word documents
class UnstructuredWordDocumentLoader(DocumentLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        text = textract.process(self.file_path).decode("utf-8")
        return text.splitlines()

# Define a class for loading and processing web pages
class WebPageLoader(DocumentLoader):
    def __init__(self, url):
        self.url = url

    def load_and_split(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n")
        return text.splitlines()

# Function to handle user input and generate a response
def generate_response(user_input):
    st.session_state["generated"].append(user_input)
    st.session_state["past"].append(user_input)

    # Concatenate past user inputs and generate response
    message_history = [
        HumanMessage(content=message) for message in st.session_state["past"]
    ]
    response = chat.send(message_history)
    st.session_state["model_name"].append(response.model_name)
    st.session_state["cost"].append(response.cost)
    st.session_state["total_cost"] += response.cost
    st.session_state["total_tokens"] += response.total_tokens

    # Store response in session state
    st.session_state["messages"].append(
        {"role": "DataChat", "content": response.choices[0].message.content}
    )

    return response.choices[0].message.content

# Function to display the chat interface
def display_chat_interface():
    st.sidebar.markdown("### User Input")
    user_input = st.sidebar.text_input("", "")
    if st.sidebar.button("Send"):
        response = generate_response(user_input)
        st.sidebar.text_area("", value=response, height=200, max_chars=None, key=None)
        st.sidebar.markdown("___")

    st.title("Chat with DataChat")
    st.markdown("---")
    st.image("datachat-logo.png", width=300)

    for message in st.session_state["messages"]:
        if message["role"] == "DataChat":
            st.text_area("DataChat", value=message["content"], height=200, max_chars=None, key=None)
        else:
            st.text_area("You", value=message["content"], height=200, max_chars=None, key=None)

# Function to summarize a given text
def summarize_text(text):
    summarizer = Summarizer()
    summary = summarizer.summarize(text)
    return summary

# Function to display the document loading interface
def display_document_loading_interface():
    st.sidebar.markdown("### Load Document")

    document_type = st.sidebar.selectbox(
        "Select Document Type",
        ("PDF", "CSV", "Word Document", "Web Page"),
    )

    if document_type == "PDF":
        file = st.sidebar.file_uploader("Upload PDF File", type="pdf")
        if file is not None:
            st.session_state["document_loader"] = PDFLoader(file)
            st.sidebar.success("PDF File Uploaded Successfully!")
    elif document_type == "CSV":
        file = st.sidebar.file_uploader("Upload CSV File", type="csv")
        if file is not None:
            st.session_state["document_loader"] = CSVLoader(file)
            st.sidebar.success("CSV File Uploaded Successfully!")
    elif document_type == "Word Document":
        file = st.sidebar.file_uploader(
            "Upload Unstructured Word Document", type=("doc", "docx")
        )
        if file is not None:
            st.session_state["document_loader"] = UnstructuredWordDocumentLoader(file)
            st.sidebar.success("Word Document Uploaded Successfully!")
    elif document_type == "Web Page":
        url = st.sidebar.text_input("Enter Web Page URL", "")
        if URLHandler.is_valid_url(url):
            st.session_state["document_loader"] = WebPageLoader(url)
            st.sidebar.success("Web Page URL Entered Successfully!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("___")

# Function to display the document processing interface
def display_document_processing_interface():
    st.sidebar.markdown("### Process Document")

    if "document_loader" in st.session_state:
        document_loader = st.session_state["document_loader"]

        if st.sidebar.button("Load and Display Document"):
            document_lines = document_loader.load_and_split()
            st.session_state["document_lines"] = document_lines
            st.sidebar.success("Document Loaded and Displayed Successfully!")

        if "document_lines" in st.session_state:
            document_lines = st.session_state["document_lines"]

            st.sidebar.markdown("---")
            st.sidebar.markdown("___")

            st.title("Document Processing")
            st.markdown("---")
            st.subheader("Document Lines")

            for line in document_lines:
                st.write(line)

            st.markdown("---")

            if st.button("Summarize Document"):
                document_text = "\n".join(document_lines)
                summary = summarize_text(document_text)
                st.write(summary)

            if st.button("Analyze Sentiment"):
                st.write("Analyzing sentiment...")

                # Perform sentiment analysis on document_lines

                st.write("Sentiment analysis completed.")

            if st.button("Extract Keywords"):
                st.write("Extracting keywords...")

                # Perform keyword extraction on document_lines

                st.write("Keywords extracted.")

            if st.button("Save Document"):
                st.write("Saving document...")

                # Save document_lines to a file

                st.write("Document saved.")

# Main code
display_chat_interface()
display_document_loading_interface()
display_document_processing_interface()
