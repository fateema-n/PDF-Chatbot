# PDF-Chatbot
This project implements a document-based chatbot application using Streamlit. Users can upload PDFs, and the chatbot retrieves relevant information based on the uploaded content.

Key Functionalities:

- PDF Processing: Extracts text from uploaded PDFs.
- Document Embedding: Generates vector representations (embeddings) for the extracted text using a pre-trained sentence transformer model.
- Pinecone Integration: Stores the document embeddings in a Pinecone vector database for efficient retrieval.
- User Interaction: Provides a Streamlit interface for users to interact with the chatbot by asking questions.

Inference:

- Queries the Pinecone index to find documents most relevant to the user's query based on their embeddings.
- Utilizes the Groq Large Language Model (LLM) API to generate a response that incorporates the retrieved information from the PDFs and provides contextually relevant answers.
