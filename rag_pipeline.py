from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# Load PDF
loader = PyPDFLoader(r"C:\Users\varsh\OneDrive\Desktop\Riya Gupta\5questions.pdf")
documents = loader.load()

# Use Ollama embedding model (lightweight)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create Vector Store
db = Chroma.from_documents(documents, embeddings)

# Use TinyLlama (lightweight model)
llm = Ollama(model="tinyllama")

# QA Chain
chain = load_qa_chain(llm, chain_type="stuff")

# Ask a query
query = "What is the assignment about?"
docs = db.similarity_search(query)
response = chain.run(input_documents=docs, question=query)

print("Answer:", response)
