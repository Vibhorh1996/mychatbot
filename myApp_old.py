# Importing libraries
import streamlit as st
import pandas as pd
import openai
import os
import json
import pickle
import requests
import pypdf
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from openai import OpenAIError
import faiss
from langchain.vectorstores import FAISS as BaseFAISS

# Class definitions

class DocumentLoader(ABC):
    """An abstract class for loading and processing documents."""

    @abstractmethod
    def load(self, file):
        """Load a file and return a pandas dataframe."""
        pass

    @abstractmethod
    def process(self, df):
        """Process a dataframe and return a list of documents."""
        pass

# Importing the pypdf library
from pypdf import PdfReader

# Defining the PDFLoader class
class PDFLoader(DocumentLoader):
    """A class for loading and processing PDF files."""

    def load(self, file):
        """Load a file and return a pandas dataframe."""
        reader = PdfReader(file) # Create a PdfReader object
        pages = reader.pages # Get the pages of the PDF file
        data = [] # Store the data
        for i, page in enumerate(pages): # Iterate over the pages
            text = page.extract_text() # Extract the text of the page
            data.append((i+1, text)) # Append the page number and text to the data
        df = pd.DataFrame(data, columns=["page", "text"]) # Create a dataframe from the data
        return df # Return the dataframe

    def process(self, df):
        """Process a dataframe and return a list of documents."""
        documents = [] # Store the documents
        for i, row in df.iterrows(): # Iterate over the rows of the dataframe
            page = row["page"] # Get the page number
            text = row["text"] # Get the text
            document = f"Page {page}:\n{text}" # Format the document
            documents.append(document) # Append the document to the list
        return documents # Return the list of documents

class FAISS(BaseFAISS):
    """A class for creating and querying a FAISS index."""

    def __init__(self):
        """Initialize the FAISS index."""
        self.index = faiss.IndexFlatL2(768) # L2 norm for cosine similarity
        self.embeddings = [] # Store the document embeddings
        self.documents = [] # Store the document texts

    def add(self, embeddings, documents):
        """Add embeddings and documents to the index."""
        self.index.add(embeddings) # Add embeddings to the index
        self.embeddings.extend(embeddings) # Extend the embeddings list
        self.documents.extend(documents) # Extend the documents list

    def search(self, query, k=5):
        """Search the index with a query and return the top k results."""
        query_embedding = self.get_query_embedding(query) # Get the query embedding
        distances, indices = self.index.search(query_embedding, k) # Search the index
        results = [] # Store the results
        for i in range(len(indices[0])):
            idx = indices[0][i] # Get the index of the result
            dist = distances[0][i] # Get the distance of the result
            doc = self.documents[idx] # Get the document text of the result
            results.append((doc, dist)) # Append the result tuple
        return results
    
    user_input = None # Define user_input as a global variable

    def ask_user():
        global user_input # Use the global keyword
        user_input = input("Make a selection: ") # Assign a value to user_input
        print("you entered", user_input)
        return user_input

    
    def get_query_embedding(self, query):
        try:
            response = openai.Completion.create(
            prompt=f"Question: {user_input}\n\nDocuments:\n{faiss_index.documents}\n\nAnswer:",
            model=model,
            return_metadata=True
        )
            answer = response["choices"][0]["text"] # Get the answer
            tokens = response["metadata"]["tokens"] # Get the number of tokens
            cost = tokens * 0.00006 # Calculate the cost
            query_embedding = response["query_embedding"]  # Get the query embedding
            return query_embedding  # Return the query embedding
        except OpenAIError as e:
            print(e)  # Print the error

# URL handler

class URLHandler:
    """A class for handling URLs."""

    @staticmethod
    def is_valid_url(url):
        """Check if a URL is valid."""
        try:
            response = requests.get(url) # Send a GET request to the URL
            return response.status_code == 200 # Return True if the status code is 200 (OK)
        except:
            return False # Return False otherwise

    @staticmethod
    def extract_links(url):
        """Extract links from a website."""
        links = [] # Store the links
        response = requests.get(url) # Send a GET request to the URL
        soup = BeautifulSoup(response.text, "html.parser") # Parse the HTML with BeautifulSoup
        for link in soup.find_all("a"): # Find all the anchor tags
            href = link.get("href") # Get the href attribute of the tag
            if href and href.endswith(".pdf"): # Check if the href is not None and ends with .pdf
                links.append(href) # Append the link to the list
        return links

# Setting page configurations

st.set_page_config(
    page_title="Data Chat", # Set the page title
    page_icon="🗣️", # Set the page icon
    layout="wide" # Set the layout to wide
)

st.markdown("# Data Chat") # Set the header

# API key configuration

st.markdown("## OpenAI API Key") # Set the subheader

api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password") # Get the API key from the user

if api_key: # Check if the API key is not empty
    os.environ["OPENAI_API_KEY"] = api_key # Set the API key as an environment variable
    import openai # Import the OpenAI library
    st.success("API key set successfully") # Display a success message
else:
    st.error("Please enter a valid API key") # Display an error message

# Session state variables

if "history" not in st.session_state: # Check if the history variable is not in the session state
    st.session_state.history = [] # Initialize the history variable as an empty list

if "model" not in st.session_state: # Check if the model variable is not in the session state
    st.session_state.model = "gpt-3.5" # Initialize the model variable as gpt-3.5

if "cost" not in st.session_state: # Check if the cost variable is not in the session state
    st.session_state.cost = 0.0 # Initialize the cost variable as 0.0

if "tokens" not in st.session_state: # Check if the tokens variable is not in the session state
    st.session_state.tokens = 0 # Initialize the tokens variable as 0

# Sidebar

st.sidebar.markdown("## Model Selection") # Set the subheader

model = st.sidebar.selectbox( # Create a selectbox for choosing the model
    "Choose a model",
    ("gpt-3.5", "gpt-4")
)

st.sidebar.markdown("## Conversation Cost") # Set the subheader

st.sidebar.write(f"Total cost: ${st.session_state.cost:.2f}") # Display the total cost
st.sidebar.write(f"Total tokens: {st.session_state.tokens}") # Display the total tokens

st.sidebar.markdown("## Clear History") # Set the subheader

clear = st.sidebar.button("Clear") # Create a button for clearing the history

if clear: # Check if the button is clicked
    st.session_state.history = [] # Reset the history variable to an empty list
    st.session_state.cost = 0.0 # Reset the cost variable to 0.0
    st.session_state.tokens = 0 # Reset the tokens variable to 0
    st.experimental_rerun() # Rerun the app

# File uploader

st.markdown("## File Uploader") # Set the subheader

file = st.file_uploader( # Create a file uploader widget
    "Upload a file",
    type=["pdf", "csv"]
)

if file: # Check if a file is uploaded
    file_name = file.name # Get the file name
    file_type = file.type # Get the file type
    file_size = file.size # Get the file size
    st.write(f"File name: {file_name}") # Display the file name
    st.write(f"File type: {file_type}") # Display the file type
    st.write(f"File size: {file_size} bytes") # Display the file size

    if file_type == "application/pdf": # Check if the file type is PDF
        faiss_index = FAISS() # Create an instance of FAISS class
        pdf_loader = PDFLoader() # Create an instance of PDFLoader class
        df = pdf_loader.load(file) # Load the file and get a dataframe
        documents = pdf_loader.process(df) # Process the dataframe and get a list of documents
        # Generate embeddings for the documents using OpenAI
        embeddings = []
        for document in documents:
            embedding = faiss_index.get_query_embedding(document)
            embeddings.append(embedding)

        # Add the embeddings and documents to the FAISS index
        faiss_index.add(embeddings, documents)

        # Display a success message
        st.success("PDF file loaded and indexed successfully")

    elif file_type == "text/csv": # Check if the file type is CSV
        csv_loader = CSVLoader() # Create an instance of CSVLoader class
        df = csv_loader.load(file) # Load the file and get a dataframe
        documents = csv_loader.process(df) # Process the dataframe and get a list of documents
        faiss_index = FAISS() # Create an instance of FAISS class
        faiss_index.add(documents) # Add the documents to the FAISS index
        st.success("CSV file loaded and indexed successfully") # Display a success message

# faiss_index = FAISS() # Create an instance of FAISS class
# faiss_index.add(documents) # Add the documents to the FAISS index
# st.success("PDF file loaded and indexed successfully") # Display a success message

# Chat interface

st.markdown("## Chat Interface") # Set the subheader

container = st.container() # Create a container for the chat interface

user_input = st.text_area("Enter your message") # Create a text area for user input

submit = st.button("Submit") # Create a button for submitting user input

if submit: # Check if the button is clicked
    if user_input: # Check if the user input is not empty
        st.session_state.history.append(("User", user_input))        # Generate a response using the selected model
        try:
            response = openai.Answer.create(
                question=user_input,
                documents=faiss_index.documents,
                model=model,
                return_metadata=True
            )
            answer = response["answer"] # Get the answer
            tokens = response["metadata"]["tokens"] # Get the number of tokens
            cost = tokens * 0.00006 # Calculate the cost
            st.session_state.history.append((model, answer)) # Append the model and answer to the history
            st.session_state.tokens += tokens # Update the tokens
            st.session_state.cost += cost # Update the cost
        except OpenAIError as e:
            print(e) # Print the error

    else:
        st.error("Please enter a valid message") # Display an error message

# Display the conversation history and cost
for speaker, message in st.session_state.history: # Iterate over the history
    if speaker == "User": # Check if the speaker is User
        container.write(f"> {message}") # Display the user message with a > prefix
    else: # Otherwise
        container.write(f"{speaker}: {message} ({tokens} tokens, ${cost:.2f})") # Display the model name, answer, tokens, and cost

container.write(f"Total cost: ${st.session_state.cost:.2f}") # Display the total cost
container.write(f"Total tokens: {st.session_state.tokens}") # Display the total tokens
