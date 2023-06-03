# Importing the required libraries
import streamlit as st
import pandas as pd
import requests
import pickle
from langchain.document_loaders import PyPDFLoader
import openai
from bs4 import BeautifulSoup
import tempfile
from openai import api_key
from tenacity import retry, stop_after_attempt, wait_exponential

# Setting the OpenAI API key
api_key = st.text_input("Please enter your OpenAI API key",'',type = 'password')
openai.api_key = api_key

# # Creating a list of available language models
# models = ["ada", "babbage", "curie", "davinci"]

# # Creating a dropdown menu for model selection
# model = st.selectbox("Please select a language model", models)

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

# Creating a file uploader for PDF files
uploaded_files = st.file_uploader("Please upload PDF files", type="pdf", accept_multiple_files=True)

# Creating an empty list to store the PDF text
pdf_text = []

# Parsing the PDF files and extracting the text
if uploaded_files:
    for file in uploaded_files:
        # Creating a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Writing the contents of the UploadedFile object to the temporary file
            temp_file.write(file.read())
            # Getting the file path of the temporary file
            temp_file_path = temp_file.name
        # Passing the file path to the PyPDFLoader class
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        for page in pages:
            text = page.page_content
            pdf_text.append(text)

# Creating a text input for user query
query = st.text_input("Please enter your query")

# Creating an empty string to store the response
response = ""

# Creating a variable to store the total cost of the conversation
total_cost = 0

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=5, max=15), reraise=True)

# Defining a function to generate responses using the OpenAI API
def generate_response(query, model):
    global response
    global total_cost

    # Pre-processing the query text
    query = query.lower()
    query = query.strip()

    # Checking if the query is empty or not
    if query == "":
        response = "Please enter a valid query"
        return

    # Creating the prompt text by concatenating the PDF text and the instructions
    prompt = "\n".join(pdf_text)
    prompt += "\n\n'''"
    prompt += "\nI want you to act as a documents that I am having a conversation with. Your name is 'AI Assistant'. You will provide me with answers from the given info. If the answer is not included, say exactly 'Hmm, I am not sure.' and stop after that. Refuse to answer any question not about the info. Never break character."
    prompt += "\n'''"

    # Appending the user query to the prompt text
    prompt += "\n\nUser: " + query + "\nAI Assistant:"
    
    # Making an API request to generate a response
    result = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        stop="\n",
        temperature=0.5,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        logprobs=10,
        echo=False)
        #return_metadata=True,

    # Extracting the response text from the result object
    response = result["choices"][0]["text"]

    # Extracting the cost of the response from the result object
    cost = result["metadata"]["cost"]

    # Updating the total cost of the conversation
    total_cost += cost

# Calling the generate_response function with the user query and model as arguments
generate_response(query, model)

# Displaying the response to the user
st.write("AI Assistant:", response)

# Displaying the total cost of the conversation
st.write("Total cost of this conversation:", total_cost)
# Creating a text input for web scraping
url = st.text_input("Please enter a URL to scrape information from")

# Creating an empty string to store the scraped content
scraped_content = ""

# Defining a function to scrape information from a web page
def scrape_info(url):
    global scraped_content

    # Checking if the URL is empty or not
    if url == "":
        scraped_content = "Please enter a valid URL"
        return

    # Making an HTTP GET request to the URL
    response = requests.get(url)

    # Checking if the response status code is 200 or not
    if response.status_code != 200:
        scraped_content = "The URL is not accessible"
        return

    # Parsing the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extracting the title of the web page
    title = soup.find("title").text

    # Extracting the text content of the web page
    text = soup.get_text()

    # Concatenating the title and text content
    scraped_content = title + "\n" + text

# Calling the scrape_info function with the URL as argument
scrape_info(url)

# Displaying the scraped content to the user
st.write("Scraped content:", scraped_content)

# Creating a text input for token counting
text = st.text_input("Please enter a text string to count tokens")

# Creating a variable to store the number of tokens
num_tokens = 0

# Defining a function to count tokens using a language model
def count_tokens(text, model):
    global num_tokens

    # Checking if the text is empty or not
    if text == "":
        num_tokens = 0
        return

    # Making an API request to encode the text using a language model
    result = openai.Encodings.create(
        engine=model,
        query=text,
        max_tokens=100,
        echo=False,
        return_metadata=True,
    )

    # Extracting the number of tokens from the result object
    num_tokens = result["metadata"]["tokens"]

# Calling the count_tokens function with the text and model as arguments
count_tokens(text, model)

# Displaying the number of tokens to the user
st.write("Number of tokens:", num_tokens)
