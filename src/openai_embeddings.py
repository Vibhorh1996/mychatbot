import openai
from tokenizer.tokenizer import Tokenizer

class OpenAIEmbeddings:
    def __init__(self, model_name, openai_api_key):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.tokenizer = openai.ChatCompletion.create(model=model_name, messages=messages)
        self.encoder = openai.TextEmbeddings.create(model=model_name)
    
    def embed(self, text):
        inputs = self.tokenizer.encode(text)
        embedding = openai.Embed(inputs)
        return embedding


class ChatOpenAI:
    def __init__(self, model_name, openai_api_key):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.chat_model = openai.ChatCompletion.create(model=model_name)
   
    def generate_response(self, input_text):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
        response = self.chat_model.create(messages=messages)
        return response.choices[0].message.content

    
#     def generate_response(self, input_text):
#         response = self.chat_model.create(
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": input_text}
#             ]
#         )
#         return response.choices[0].message.content
