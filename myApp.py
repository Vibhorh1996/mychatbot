import streamlit as st
import pandas as pd
from llama_index import GPTPandasIndex
import os

st.write(' # JazzHR')
st.write('A HR Analytics Conversational AI')

os.environ['OPENAI_API_KEY'] = 'your key'

uploaded_file = st.file_uploader("Choose a CSV file",accept_multiple_files=False)
#bytes_data = uploaded_file.read()
try:
    #st.write("filename:", uploaded_file.name)
    st.write(uploaded_file)
    df = pd.read_csv(uploaded_file)
    st.write(df)

    index = GPTPandasIndex.from_dataframe(df)

    while True:
        prompt = input("write your question:")
        response = index.query(prompt)
        st.write(response)

        # Get the last token usage
        last_token_usage = index.llm_predictor.last_token_usage
        st.write(f"last_token_usage={last_token_usage}")


except :
    st.write("No file uploaded yet")

#create a try except block
