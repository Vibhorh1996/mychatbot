import openai

class Embed:
    def __init__(self, model, inputs, engine="davinci-codex", prompt_label="text"):
        self.model = model
        self.inputs = inputs
        self.engine = engine
        self.prompt_label = prompt_label
        self.embeddings = None

    @classmethod
    def create(cls, model, inputs, engine="davinci-codex", prompt_label="text"):
        embed = cls(model, inputs, engine, prompt_label)
        embed.create_embeddings()
        return embed

    def create_embeddings(self):
        response = openai.EmbeddingCompletion.create(
            model=self.model,
            inputs=self.inputs,
            engine=self.engine,
            prompt_label=self.prompt_label
        )
        self.embeddings = response.choices[0].doc_embeddings

    def get_embeddings(self):
        return self.embeddings
