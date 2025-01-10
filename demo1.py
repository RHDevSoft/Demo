import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI
import base64
import streamlit.components.v1 as components
import asyncio
import re
import hashlib
import json
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

PERSIST_DIR = "./db"
DATA_DIR = "data"
CHUNK_SIZE = 100  # Number of pages to process at a time

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def display_pdf(file):
    """Display PDF in Streamlit app."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def sanitize_category_name(name):
    """Sanitize category name to be a valid and short directory name."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)  # Remove invalid characters
    return hashlib.md5(name.encode()).hexdigest()  # Use MD5 hash to generate a unique directory name

def load_category_mapping():
    """Load category mapping from JSON file."""
    try:
        with open(os.path.join(PERSIST_DIR, "categories.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_category_mapping(mapping):
    """Save category mapping to JSON file."""
    with open(os.path.join(PERSIST_DIR, "categories.json"), "w") as f:
        json.dump(mapping, f)

def categorize_documents(doc_pages):
    """Categorize documents using LLM."""
    categories = {}
    category_mapping = load_category_mapping()
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    for page in doc_pages:
        logging.debug(f"Processing page content: {page.page_content[:100]}...")
        prompt = f"Provide a single, concise category name for the following text:\n\n{page.page_content.strip()}"
        response = llm(prompt)
        category = response.content.strip()
        sanitized_category = sanitize_category_name(category)
        
        if sanitized_category not in category_mapping:
            category_mapping[sanitized_category] = category
            logging.info(f"New category added: {category} with sanitized version: {sanitized_category}")
        
        if sanitized_category not in categories:
            categories[sanitized_category] = []
        categories[sanitized_category].append(page)
        
        st.write(f"Original category: '{category}', Sanitized category: '{sanitized_category}'")

    save_category_mapping(category_mapping)
    return categories

async def data_ingestion():
    """Ingest data from PDF files and create a vector store."""
    total_files = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
    progress_bar = st.progress(0)
    
    for idx, filename in enumerate(os.listdir(DATA_DIR)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyMuPDFLoader(filepath)
            doc_pages = loader.load()

            # Categorize documents using LLM
            categories = categorize_documents(doc_pages)

            # Initialize HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

            # Create a FAISS vector store for each category
            for category, docs in categories.items():
                category_path = os.path.join(PERSIST_DIR, category)
                os.makedirs(category_path, exist_ok=True)
                vector_store = FAISS.from_documents(docs, embeddings)
                vector_store.save_local(category_path)
                st.write(f"Saved category '{category}' with {len(docs)} documents at '{category_path}'.")

        # Update progress bar
        progress_bar.progress(min((idx + 1) / total_files, 1.0))

    st.success("Data ingestion completed successfully!")

async def handle_query(query):
    """Handle user query and generate a response."""
    try:
        category_mapping = load_category_mapping()
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Determine the category based on the query using LLM
        prompt = f"Determine the category for the following query:\n\n{query}"
        response = llm(prompt)
        category = response.content.strip()
        sanitized_category = sanitize_category_name(category)
        
        st.write(f"Original category: '{category}', Sanitized category: '{sanitized_category}'")

        # Check if category exists in the mapping
        if sanitized_category in category_mapping:
            category_path = os.path.join(PERSIST_DIR, sanitized_category)
            if not os.path.exists(category_path):
                st.error(f"Category '{sanitized_category}' not found in storage.")
                return "Sorry, I couldn't find information related to your query."
            
            # Load the vector store
            vector_store = FAISS.load_local(
                category_path,
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
            )
            
            # Retrieve relevant documents
            docs = vector_store.similarity_search(query)

            # Combine the content of the retrieved documents
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = ""
            st.warning(f"Category '{sanitized_category}' not found in the mapping. Providing a general response.")

        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are an intelligent assistant named Chat-bot. Your main goal is to provide answers based on the content of provided PDFs. 
            If the information is not found in the PDFs, answer as best as possible based on your general knowledge, but prioritize PDF content.
            Context:
            {context}
            Question:
            {query}
            """
        )

        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            callbacks=[StreamlitCallbackHandler(st.container())]
        )
        
        response = await chain.arun(context=context, query=query)
        analyze_interactions(query, response)
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, an error occurred while processing your query."

def analyze_interactions(query, response):
    """Analyze user interactions and provide insights or recommendations."""
    # Example implementation: Log interactions to a file (can be extended to provide more advanced insights)
    with open("interactions_log.txt", "a") as log_file:
        log_file.write(f"Query: {query}\nResponse: {response}\n\n")

st.title("EXL Chatbot🤖")
st.markdown("Created by Pratap") 

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Please upload a PDF and ask a question.'}]

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    if uploaded_files and st.button("Submit"):
        with st.spinner("Processing..."):
            for uploaded_file in uploaded_files:
                filepath = os.path.join(DATA_DIR, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            asyncio.run(data_ingestion())  
            st.success("Done")

user_prompt = st.chat_input("Ask me")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = asyncio.run(handle_query(user_prompt))
    st.session_state.messages.append({'role': 'assistant', "content": response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
