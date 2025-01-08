import os
import asyncio
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Load environment variables
load_dotenv()

PERSIST_DIR = "./db"
DATA_DIR = "data"
CHUNK_SIZE = 100  # Number of pages to process at a time

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def display_pdf(file):
    """Displays PDF file in Streamlit."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

async def data_ingestion():
    """Ingests and processes PDF files into vector stores."""
    pdf_data = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                loader = PyMuPDFLoader(filepath)
                doc_pages = loader.load()

                # Log the number of pages loaded from the PDF
                st.write(f"Loaded {len(doc_pages)} pages from {filename}")

                # Process in chunks
                documents = []
                for i in range(0, len(doc_pages), CHUNK_SIZE):
                    chunk = doc_pages[i:i + CHUNK_SIZE]
                    documents.extend(chunk)

                # Initialize embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vector_store = FAISS.from_documents(documents, embeddings)
                vector_store.save_local(os.path.join(PERSIST_DIR, filename))
                pdf_data[filename] = vector_store
            except Exception as e:
                st.error(f"Failed to process {filename}: {e}")

    st.session_state.pdf_data = pdf_data
    st.success("Data ingestion completed successfully!")

async def process_query_with_best_match(query):
    """Processes user query and returns the best matching documents."""
    try:
        best_match_filename = None
        best_score = float('-inf')
        best_docs = None

        # Search across all PDFs to find the best match
        for filename, vector_store in st.session_state.pdf_data.items():
            vector_store_path = os.path.join(PERSIST_DIR, filename)

            # Load the vector store with allow_dangerous_deserialization=True to bypass the security warning
            vector_store = FAISS.load_local(
                vector_store_path, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True  # Allow dangerous deserialization (make sure it's safe)
            )
            
            # Ensure the vector store has documents
            if vector_store:
                # Perform similarity search
                docs = vector_store.similarity_search(query, top_k=5)

                # Log to inspect the results
                st.write(f"Found {len(docs)} documents for query '{query}' from {filename}")

                # Calculate score based on content length and match percentage
                score = len(docs) if docs else 0

                if score > best_score:
                    best_score = score
                    best_match_filename = filename
                    best_docs = docs

        if best_docs:
            context = "\n".join([doc.page_content for doc in best_docs])

            st.write(f"Best match from {best_match_filename}: {context[:500]}...")  # Show preview

            # Initialize the Azure OpenAI client and process the response
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

            # Prepare the prompt
            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""Given the following context, answer the user's question in a clear and concise manner.

                Context:
                {context}

                User Query:
                {query}
                
                Provide a detailed, accurate, and structured answer based on the context."""
            )

            formatted_prompt = prompt_template.format(context=context, query=query)

            # Initialize the tool for document retrieval
            tool = Tool(
                name="Document Retrieval",
                func=lambda x: context,
                description="Retrieve context for the query"
            )

            # Initialize the agent chain
            agent_chain = initialize_agent(
                [tool], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
            )

            # Run the agent and get the response
            response = await agent_chain.arun(formatted_prompt, handle_parsing_errors=True)

            st.write(f"Agent response: {response}")

        else:
            st.write("No relevant context found across the uploaded PDFs.")

    except Exception as e:
        st.error(f"An error occurred while processing the query: {e}")

# Streamlit app for file upload and querying
st.title("PDF Processing Chatbot")
st.markdown("Upload PDFs and ask questions based on their contents.")

if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = {}

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    if uploaded_files and st.button("Submit"):
        with st.spinner("Processing..."):
            for uploaded_file in uploaded_files:
                filepath = os.path.join(DATA_DIR, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            asyncio.run(data_ingestion())

# Process user query and search for best match across PDFs
query = st.text_input("Ask a question based on the uploaded PDFs:")
if query:
    with st.spinner("Processing your query..."):
        asyncio.run(process_query_with_best_match(query))
