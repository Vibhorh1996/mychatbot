import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import os
import re
import json
import openai

# Set your OpenAI API key here
#OPENAI_API_KEY = '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'

def generate_response(input_text, model='gpt-3.5-turbo', max_tokens=50):
    api = OpenAIApi(OPENAI_API_KEY)
    response = api.complete(
        prompt=input_text,
        model=model,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.6,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        log_level='info'
    )
    return response['choices'][0]['text'].strip()

def run_data_chat():
    # Set Streamlit app title
    st.title("Data Chat")

    # Display app description and instructions
    st.markdown("Data Chat is a chatbot application that allows users to interact with a pre-trained language model.")
    st.markdown("To get started, please provide your OpenAI API key:")

    # Get OpenAI API key from user input
    openai_api_key = st.text_input("OpenAI API Key")

    # Set OpenAI API key in environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Check if API key is provided
    if not openai_api_key:
        st.warning("Please provide your OpenAI API key to proceed.")
        return

    # Language model selection
    models = {
        'gpt-3.5-turbo': 'GPT-3.5 Turbo',
        # Add more models here if desired
    }

    selected_model = st.selectbox("Select a language model", list(models.values()))

    # Set the chosen model for generating responses
    model = next(key for key, value in models.items() if value == selected_model)

    # Start the conversation with the AI assistant
    st.markdown("## Start a conversation with AI Assistant")

    conversation_history = []

    while True:
        user_input = st.text_input("User:", key='user_input')

        if st.button("Ask"):
            # Add user input to conversation history
            conversation_history.append({'role': 'user', 'content': user_input})

            # Generate response from the AI assistant
            response = generate_response('\n'.join([f"{entry['role']}: {entry['content']}" for entry in conversation_history]),
                                         model=model)

            # Add AI assistant response to conversation history
            conversation_history.append({'role': 'assistant', 'content': response})

            # Display AI assistant response
            st.text_area("AI Assistant:", value=response, height=100, key='response')

        if st.button("Reset Conversation"):
            # Clear conversation history
            conversation_history = []

        if st.button("End Conversation"):
            # Clear conversation history and display conversation cost
            conversation_tokens = sum([len(entry['content'].split()) for entry in conversation_history])
            st.info(f"Conversation Tokens: {conversation_tokens}")
            st.info(f"Conversation Cost: {conversation_tokens * 0.004:.2f} USD")

            # Clear conversation history
            conversation_history = []

def upload_files():
    uploaded_files = st.file_uploader('Upload PDF Files', type='pdf', accept_multiple_files=True)
    return uploaded_files
uploaded_files = upload_files()

def parse_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
parsed_texts = []
for file in uploaded_files:
    text = parse_pdf(file)
    parsed_texts.append(text)
            
if __name__ == '__main__':
    run_data_chat()
