# Simple RAG
A simple implementation of a RAG pipeline using Pinecone as a vector database and ollama as a chatbot.

## Configuration
`config.yaml` contains some default values for embedding and language models that run this simple RAG implementation. These can be edited for your specific use case.

## Usage
Create a RAG object:

```
from rag import RAG
r = RAG()
```

Create the vector embeddings. Documents should be an array of dictionaries:

```
documents = [
    {"id": "foo", "text": "bar"},
    ...
]
r.create_embeddings(documents)
```

Query the database directly:

```
query = "this is a query"

# simple retrieval
results = r.retrieve(query)
print(results)

# retrieval with reranking
retrieved_documents = r.retrieve(query)
results = r.retrieve_with_rerank(query, retrieved_documents)
print(results)
```

Or, if not querying the database directly, talk to the chatbot:

```
from bot import ChatBot
b = ChatBot()
b.chat()
```

See the include Python Notebook for code use examples. There are 2 ways you can use this simple RAG pipeline:

- directly creating a RAG object and calling `retrieve` or `retrieve_with_rerank` to get the raw RAG responses.
- creating a ChatBot object and calling `chat` to interact with the ollama chatbot.

## Resources
The following links were used to help create this simple RAG pipeline:

- https://huggingface.co/blog/ngxson/make-your-own-rag
- https://www.pinecone.io/learn/series/rag/rerankers/