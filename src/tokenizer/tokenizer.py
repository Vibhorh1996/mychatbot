import openai

class Tokenizer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = Tokenizer(model=model_name)
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
