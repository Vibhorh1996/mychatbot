import openai

class Embed:
    def __init__(self, inputs, engine="davinci-codex", prompt_label="text"):
        #self.model = model
        self.inputs = inputs
        self.engine = engine
        self.prompt_label = prompt_label
        self.embeddings = None

    @classmethod
    def create(cls, inputs, engine="davinci-codex", prompt_label="text"):
        embed = cls(inputs, engine, prompt_label)
        embed.create_embeddings()
        return embed

    def create_embeddings(self):
        response = openai.Completion.create(
            #model=self.model removed model param from here and from all other functions where it was taken as a param. init,create
            inputs=self.inputs,
            engine=self.engine,
            prompt_label=self.prompt_label,
            embed=True
        )
        self.embeddings = response.choices[0].doc_embeddings

    def get_embeddings(self):
        return self.embeddings
