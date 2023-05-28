import streamlit as st
from streamlit_chat import message
import pandas as pd
import os
import json
import pickle
from abc import ABC, abstractmethod
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
import tiktoken
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as BaseFAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.document_loaders import WebBaseLoader

# Helper functions
def is_valid_pdf(file_path):
    """
    Checks if the file at the given path is a valid PDF file.
    """
    return file_path.lower().endswith(".pdf")

def extract_links_from_pdf(file_path):
    """
    Extracts links from a PDF file using a custom logic.
    Modify this function if you have a specific requirement for link extraction from PDFs.
    """
    # Placeholder implementation
    return []

def extract_links_from_files(file_paths):
    """
    Extracts links from multiple PDF files.
    """
    all_links = []

    for file_path in file_paths:
        if is_valid_pdf(file_path):
            links = extract_links_from_pdf(file_path)
            all_links.extend(links)

    return all_links

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
        # Custom logic for link extraction from URLs
        pass

    @staticmethod
    def extract_links_from_websites(websites):
        # Custom logic for link extraction from websites
        pass

class DataChatApp:
    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.file_paths = []
        self.index_name = "data_index"

    def load_models(self):
        # Load or train the models and indices
        self.load_faiss_index()
        self.load_chat_model()

    def load_faiss_index(self):
        if os.path.exists("models/faiss_index.pickle"):
            with open("models/faiss_index.pickle", "rb") as f:
                self.faiss_index = pickle.load(f)
        else:
            self.faiss_index = FAISS()

    def load_chat_model(self):
        if os.path.exists("models/chat_model.pickle"):
            with open("models/chat_model.pickle", "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = ChatOpenAI()

    def save_models(self):
        # Save the models and indices
        self.save_faiss_index()
        self.save_chat_model()

    def save_faiss_index(self):
        with open("models/faiss_index.pickle", "wb") as f:
            pickle.dump(self.faiss_index, f)

    def save_chat_model(self):
        with open("models/chat_model.pickle", "wb") as f:
            pickle.dump(self.model, f)

    def add_documents(self, documents):
        # Add documents to the index
        self.faiss_index.add_documents(documents)

    def answer_questions(self, questions):
        # Answer questions using the chat model
        responses = []
        for question in questions:
            response = self.model.generate_response(question)
            responses.append(response)
        return responses

    def run(self):
        self.load_models()

        st.set_page_config(page_title="Data Chat App", page_icon=":chart_with_upwards_trend:")

        st.title("Data Chat App")
        st.markdown("Welcome to the Data Chat App. Upload your PDF files to get started!")

        uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type="pdf")

        if uploaded_files:
            file_paths = []
            for file in uploaded_files:
                file_path = os.path.join("uploads", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)

            self.file_paths = file_paths

            st.success(f"Successfully uploaded {len(file_paths)} PDF files.")

            # Extract links from PDF files
            links = extract_links_from_files(file_paths)

            # Handle URLs and extract links from websites
            url_links = [link for link in links if URLHandler.is_valid_url(link)]
            website_links = [link for link in links if not URLHandler.is_valid_url(link)]

            URLHandler.extract_links_from_websites(website_links)

            # Add links to the index
            self.add_documents(links)

            st.markdown("### Ask your questions:")
            with st.form(key="question_form"):
                question_input = st.text_input("Enter your question")
                submit_button = st.form_submit_button("Ask")

            if submit_button:
                if question_input:
                    question = question_input.strip()
                    st.info(f"You: {question}")

                    # Answer the question
                    response = self.answer_questions([question])[0]

                    # Display the response
                    st.success(f"Chatbot: {response}")
                else:
                    st.warning("Please enter a question.")

        self.save_models()

if __name__ == "__main__":
    app = DataChatApp()
    app.run()

