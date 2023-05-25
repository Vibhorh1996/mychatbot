import openai
from tokenizer.tokenizer import Tokenizer

class OpenAIEmbeddings:
    def __init__(self, model_name, openai_api_key, messages=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            self.messages = messages
            self.chat_model = openai.ChatCompletion.create(model=model_name, messages = messages)
        self.tokenizer = openai.ChatCompletion.create(model=model_name, messages=self.messages)
        #self.encoder = openai.TextEmbeddings.create(model=model_name)
    
    def embed(self, text):
        inputs = self.tokenizer.encode(text)
        embedding = openai.Embed.create(
            model=self.model_name,
            inputs=inputs,
            engine="davinci-codex",
            prompt_label="text"
        )
        return embedding.choices[0].doc_embeddings
    
#     def embed(self, text):
#         inputs = self.tokenizer.encode(text)
#         embedding = openai.Embed(inputs)
#         return embedding


class ChatOpenAI:
    def __init__(self, model_name, openai_api_key, messages=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            self.chat_model = openai.ChatCompletion.create(model=model_name, messages = messages)
        self.chat_model = openai.ChatCompletion.create(model=model_name, messages=messages)
   
    def generate_response(self, input_text):
        self.messages.append({"role": "system", "content": "You are a helpful assistant."})
        self.messages.append({"role": "user", "content": input_text})
        
        while True:
            try:
                response = self.chat_model.create(messages=self.messages)
                break  # Break out of the loop if the request succeeds
            except openai.error.RateLimitError as e:
                print("Rate limit reached. Waiting for 20 seconds...")
                time.sleep(20)  # Wait for 20 seconds
                continue  # Retry the request
            
        self.messages = [{"role": "system", "content": response.choices[0].message.content}]
        return response.choices[0].message.content
    
#     def generate_response(self, input_text):
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": input_text}
#         ]
#         response = self.chat_model.create(messages=messages)
#         return response.choices[0].message.content

    
#     def generate_response(self, input_text):
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": input_text}
#         ]
#         response = self.chat_model.create(messages=messages)
#         return response.choices[0].message.content

    
#     def generate_response(self, input_text):
#         response = self.chat_model.create(
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": input_text}
#             ]
#         )
#         return response.choices[0].message.content
