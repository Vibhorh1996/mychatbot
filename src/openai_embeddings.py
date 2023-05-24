import openai

class OpenAIEmbeddings:
    def __init__(self, model_name, openai_api_key):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.tokenizer = openai.Tokenizer(model=model_name)
        self.encoder = openai.Encoder(model=model_name)
    
    def embed(self, text):
        inputs = self.tokenizer.encode(text)
        embedding = self.encoder.encode(inputs)
        return embedding


class ChatOpenAI:
    def __init__(self, model_name, openai_api_key):
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.chat_model = openai.ChatCompletion.create(model=model_name)
    
    def generate_response(self, input_text):
        response = self.chat_model.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content
