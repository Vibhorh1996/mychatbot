import streamlit as st
from streamlit_chat import message
import pandas as pd
from llama_index.indices.struct_store import GPTPandasIndex
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex
import os
import json


# setting page title and header
st.set_page_config(page_title="JazzHR", page_icon=':robot_face:')
st.markdown("<h1 stype='text-align:center;'>JazzHR</h1>", unsafe_allow_html=True)
st.markdown("<h2 stype='text-align:center;'>A HR Analytics Conversational AI </h2>", unsafe_allow_html=True)

# set API Key
key = st.text_input('OpenAI API Key','',type='password')
os.environ['OPENAI_API_KEY'] = key


# initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated']=[]

if 'past' not in st.session_state:
    st.session_state['past']=[]

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {"role":"JazzHR","content":"You are a helpful bot."}
    ]

if 'model_name' not in st.session_state:
    st.session_state['model_name']=[]

if 'cost' not in st.session_state:
    st.session_state['cost']=[]

if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens']=[]

if 'total_cost' not in st.session_state:
    st.session_state['total_cost']=0.0

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
        
# def askQuestion():
#     prompt = st.text_input("write your question:")
#     response = index.query(prompt)
#     st.write(response)

#     # Get the last token usage
#     last_token_usage = index.llm_predictor.last_token_usage
#     st.write(f"last_token_usage={last_token_usage}")

def save_uploadedfile(uploadedfile):
     with open(os.path.join("data/dataset","temp.csv"),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to dataset".format(uploadedfile.name))


def generate_response(index,prompt):
    st.session_state['messages'].append({"role":"user","content":prompt})

    response = index.query(prompt)
    st.session_state['messages'].append({"role":"JazzHR","content":response})

    #last_token_usage = index.llm_predictor.last_token_usage
    last_token_usage = 0.0
    #print(f"last_token_usage={last_token_usage}")

    return response,  last_token_usage


df=None
uploaded_file = st.file_uploader("Choose a CSV file",accept_multiple_files=False)
if uploaded_file is not None:
   file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
   df  = pd.read_csv(uploaded_file)
   st.dataframe(df.head(10))
   save_uploadedfile(uploaded_file)

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

documents = SimpleDirectoryReader('data/dataset').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)





with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, last_token_count = generate_response(index,user_input)
        #st.write(output)
        total_tokens = last_token_count
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output.response)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            #cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            cost = total_tokens * 0.002 / 1000
        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    #st.write(st.session_state['generated'])
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
