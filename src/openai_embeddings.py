import openai
from tenacity import retry, stop_after_attempt, wait_exponential
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


class ChatOpenAI:
    def __init__(self, model_name, openai_api_key, messages=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            self.chat_model = openai.ChatCompletion.create(model=model_name, messages = messages)
        self.chat_model = openai.ChatCompletion.create(model=model_name, messages=messages)
    
    
    @retry(
                stop=stop_after_attempt(3),  # Retry for a maximum of 3 attempts
                wait=wait_exponential(multiplier=1, min=2, max=5)  # Exponential backoff with a base of 2
    def generate_response(self, input_text):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text}
        ]
        response = self.chat_model.create(messages=messages)
        return response.choices[0].message.content

