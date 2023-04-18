# from llama_index import CSVReader, GPTSimpleVectorIndex

#from pathlib import Path
import os
#from llama_index import download_loader, GPTSimpleVectorIndex
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

#from SimpleCSVReader import SimpleCSVReader

#SimpleCSVReader = download_loader("SimpleCSVReader")
#loader = SimpleCSVReader()
#documents = loader.load_data(file=Path('./Methane_final.csv'))

os.environ['OPENAI_API_KEY'] = ''


# Read from a local file
# reader = CSVReader("./Methane_final.csv", delimiter=",", quotechar='"', header=1)
#index = GPTSimpleVectorIndex.from_documents(documents)
#index = GPTSimpleVectorIndex(documents)

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

while True:
    prompt = input("Show me top 5 countries with high emmissions")
    response = index.query(prompt)
    print(response)

    # Get the last token usage
    last_token_usage = index.llm_predictor.last_token_usage
    print(f"last_token_usage={last_token_usage}")

# results = index.query("Show me countries with high emmissions", top_k=5)

# # Print the results
# for result in results:
#   print(result)
