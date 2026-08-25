import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

TARGET_DOCS_URLS = [
    # --- LANGCHAIN CORE MODULES & GUIDES ---
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/how_to/recursive_text_splitter/",
    "https://python.langchain.com/docs/how_to/character_text_splitter/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/document_loaders/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/lcel/",
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/integrations/vectorstores/qdrant/",
    "https://python.langchain.com/docs/integrations/text_embedding/",
    
    # --- QDRANT CORE & ADVANCED ARCHITECTURE ---
    "https://qdrant.tech/documentation/quick-start/",
    "https://qdrant.tech/documentation/concepts/collections/",
    "https://qdrant.tech/documentation/concepts/points/",
    "https://qdrant.tech/documentation/concepts/payload/",
    "https://qdrant.tech/documentation/concepts/search/",
    "https://qdrant.tech/documentation/concepts/filtering/",
    "https://qdrant.tech/documentation/concepts/storage/",
    "https://qdrant.tech/documentation/concepts/quantization/",
    "https://qdrant.tech/documentation/concepts/hybrid-search/",
    "https://qdrant.tech/documentation/concepts/optimizer/",
    "https://qdrant.tech/documentation/concepts/distributed_deployment/",
    "https://qdrant.tech/documentation/frameworks/langchain/",
    "https://qdrant.tech/documentation/embeddings/"
]

print("1. Scraping documentation pages...")
loader = WebBaseLoader(TARGET_DOCS_URLS, continue_on_failure=True)
raw_docs = loader.load()
print(f"Successfully loaded {len(raw_docs)} documentation modules.")

print("2. Splitting text into detailed semantic chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(raw_docs)
print(f"Generated {len(chunks)} searchable chunks.")

print("3. Generating embeddings model...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("4. Connecting Qdrant client with 120s timeout...")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120.0
)

print("5. Initializing vector store instance...")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="langchain_qdrant_docs",
    embedding=embeddings
)

print("6. Uploading chunks in batches of 50...")
vector_store.add_documents(chunks, batch_size=50)

print("✅ Full documentation ingestion finished!")