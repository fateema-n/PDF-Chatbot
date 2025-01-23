import streamlit as st
import os
from vectorize import process_documents, inference

# Ensure the temp directory exists
if not os.path.exists('./temp'):
    os.makedirs('./temp')

# Streamlit UI
st.title("PDF Chatbot")

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    st.sidebar.write(f"{len(uploaded_files)} PDF(s) uploaded.")
    pdf_paths = []
    for file in uploaded_files:
        pdf_path = f"./temp/{file.name}"
        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())
        pdf_paths.append(pdf_path)
    
    for pdf_path in pdf_paths:
        process_documents(None, pdf_path)  # Pass None as we do not need model for now

# Chatbot interaction
st.header("Chat with the PDF Documents")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if user_input:

    # Get the response from the inference function
    answer = inference(user_input, st.session_state.chat_history)
    
    # Update session state chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**User:** {chat['content']}")
    else:
        st.markdown(f"**Assistant:** {chat['content']}")
