import openai

class Embed:
    def __init__(self, model, inputs,  engine="davinci-codex", prompt_label="text"): #engine="davinci-codex",
        self.model = model
        self.inputs = inputs
        self.engine = engine
        self.prompt_label = prompt_label
        self.embeddings = None

    def create_embeddings(self):
        response = Completion.create(
            model=self.model,
            inputs=self.inputs,
            engine=self.engine, #removed engine param from here and from all other functions where it was taken as a param. init,create
            prompt_label=self.prompt_label,
            embed=True
        )
        print(response)
        self.embeddings = response.choices[0].doc_embeddings
    
    @classmethod
    def create(cls, model, inputs,  engine="davinci-codex", prompt_label="text"): #engine="davinci-codex",
        embed = cls(inputs, engine, prompt_label) #engine,
        embed.create_embeddings()
        return embed

    def get_embeddings(self):
        return self.embeddings
