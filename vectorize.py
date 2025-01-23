import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from groq import Client
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client using environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
client = Client(api_key=groq_api_key)

# Set your Pinecone API key using environment variable
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key  # Set API key as environment variable

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)


# Index name and embedding dimension
index_name = "langchain-chatbot"
embedding_dimension = 384  # 'all-MiniLM-L6-v2' has a dimension of 384

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_pdfs(filenames):
    """Load and extract text from a given PDF file."""
    reader = PdfReader(filenames)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_data
def split_docs(documents, chunk_size=2048, chunk_overlap=20):
    """Split the document text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(documents)
    return [Document(page_content=chunk) for chunk in chunks]

def store_embeddings(_docs, index_name):
    """Store document embeddings into Pinecone."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = LangchainPinecone.from_documents(_docs, embeddings, index_name=index_name)
    print(f"Embeddings successfully stored in the '{index_name}' index.")


def process_documents(model, pdfs):
    """Process PDF documents: extract text, split, and store embeddings."""
    text = load_pdfs(pdfs)
    docs = split_docs(text)
    store_embeddings(docs, index_name)
    

def inference(query, chat_history):
    """Perform inference by querying Pinecone and passing it to Groq LLM."""
    try:
        start_time = time.time()
        # Create an embedding for the query
        query_embedding = embedding_model.encode(query).tolist()
        embedding_time = time.time() - start_time
        print(f"Time for embedding generation: {embedding_time:.2f} seconds")

        # Query the Pinecone index
        start_time = time.time()
        
        results = index.query(
            vector=query_embedding,
            top_k=3,  # best 3 docs to match the query
            include_metadata=True,
        )
        pinecone_time = time.time() - start_time
        print(f"Time for Pinecone query: {pinecone_time:.2f} seconds")

        # Extract matched information and sources (didn't understand this part, read it l8r)
        matched_info = []
        sources = []
        for item in results["matches"]:
            text = item["metadata"].get("text", "")  # Use default empty string if 'text' is missing
            source = item["metadata"].get("source", "Unknown Source")  # Handle missing 'source' keys
            matched_info.append(text)
            sources.append(source)

        # Combine matched information and sources
        context = f"Information: {' '.join(matched_info)}\nSources: {', '.join(sources)}"

        max_history_length = 2  # Limit to the last 2 messages for testing
        conversation_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-max_history_length:]])

        # Combine the reduced conversation context with the matched information
        full_context = f"{conversation_context}\n{context}"

        # Create a system prompt
        start_time = time.time()
        sys_prompt = (
    "You are a helpful assistant. Answer the query based on the context below. "
    "If the context doesn't provide enough information, reply: 'I don't have enough information to answer this question.'"
    f"\nContext: {full_context}\nUser's Query: {query}"
)

        # Use the Groq API to get chat completions
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": sys_prompt}
            ],
            model="llama3-8b-8192"  
        )
        llm_time = time.time() - start_time
        print(f"Time for LLM inference: {llm_time:.2f} seconds")

        total_time = embedding_time + pinecone_time + llm_time
        print(f"Total time: {total_time:.2f} seconds")

        # Return the completion content
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error during inference: {e}"

