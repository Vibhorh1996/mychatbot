import openai
from src.embeddings import Embed
from tenacity import retry, stop_after_attempt, wait_exponential
from src.tokenizer.tokenizer import Tokenizer

class OpenAIEmbeddings:
    def __init__(self, model_name, openai_api_key, messages=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.messages = messages
        self.tokenizer = Embed.create(model=model_name, inputs=self.messages)
        self.chat_model = openai.ChatCompletion.create(model=model_name, messages=self.messages)

    def embed(self, texts):
        tokenized_texts = [self.tokenizer.encode(text) for text in texts]
        embeddings = []

        for tokens in tokenized_texts:
            embedding = Embed.create(
                model=self.model_name,
                inputs=tokens,
                engine="davinci-codex",
                prompt_label="text"
            )
            embeddings.append(embedding.choices[0].doc_embeddings)

        return embeddings


class ChatOpenAI:
    def __init__(self, model_name, openai_api_key, messages=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.messages = messages
        self.chat_model = openai.ChatCompletion.create(model=model_name, messages=self.messages)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def generate_response(self, input_text):
         st.session_state['messages'].append({"role":"user","content":prompt})

        response = self.chat_model.create(messages=messages)
        st.session_state['messages'].append({"role":"DataChat","content":response})
        return response
    
#     def generate_response(self, input_text):
#         messages = self.messages + [
#             {"role": "user", "content": input_text}
#         ]
#         response = self.chat_model.create(messages=messages)
#         return response.choices[0].message.content
