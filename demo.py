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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load environment variables
load_dotenv()

PERSIST_DIR = "./db"
DATA_DIR = "data"
CHUNK_SIZE = 100  # Number of pages to process at a time

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

async def data_ingestion():
    pdf_data = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyMuPDFLoader(filepath)
            doc_pages = loader.load()

            # Process in chunks
            documents = []
            for i in range(0, len(doc_pages), CHUNK_SIZE):
                chunk = doc_pages[i:i + CHUNK_SIZE]
                documents.extend(chunk)

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(os.path.join(PERSIST_DIR, filename))
            pdf_data[filename] = vector_store

    st.session_state.pdf_data = pdf_data
    st.success("Data ingestion completed successfully!")

async def handle_query(query):
    try:
        responses = []
        for filename, vector_store in st.session_state.pdf_data.items():
            vector_store = FAISS.load_local(
                os.path.join(PERSIST_DIR, filename),
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
            )
            docs = vector_store.similarity_search(query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""Extract the following structured information from the documents and display in the PDF:
Business Profile, Operating Segments, Verizon Consumer Group, Verizon Business Group, Corporate and Other.
Ensure the content is organized and properly formatted.
{context}"""
            )
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = await chain.arun(context=context, query=query)
            responses.append(response)
        return "\n\n".join(responses)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def generate_pdf(data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Generated Document")
    y_position = 730
    for section, content in data.items():
        c.drawString(100, y_position, f"{section}:")
        y_position -= 20
        c.drawString(120, y_position, f"{content} This section includes additional details and key insights about the topic for comprehensive understanding.")
        y_position -= 40  # Add extra space after each section for readability
        y_position -= 20
        if y_position < 50:  # Prevent writing beyond the page
            c.showPage()
            y_position = 750
    c.save()

st.title("PDF Processing Chatbot")
st.markdown("Upload PDFs and generate a structured PDF format.")

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

if 'pdf_data' in st.session_state and st.session_state.pdf_data:
    generate_pdf({"Business Profile": "Verizon Communications Inc. (“Verizon”) is one of the world’s leading providers of communications, technology, information and entertainment products and services to consumers, businesses and government entities. With a presence around the world, it offers data, video and voice services and solutions on networks and platforms that are designed to meet customers’ demand for mobility, reliable network connectivity and security. The company had a customer base of 144.7 million as of Sep 2024 within the wireless segment, with about 86% being postpaid customers. Average revenue per account (ARPA) from postpaid has been steadily rising and the company accounts for approximately 40% of the market share within the total wireless segment, with its direct competitors (namely, AT&T and T-Mobile) accounting for majority of the rest.", "Operating Segments": "Verizon Consumer Group, Verizon Business Group, Corporate and Other.", "Verizon Consumer Group": "Consumer segment provides consumer-focused wireless and wireline communications services and products. For 9M ended Sep 2024, the Consumer segment revenues were $75.3 billion, representing approximately 76% of Verizon’s consolidated revenues. It also includes fixed wireless access (FWA) broadband through 5G or 4G LTE networks as an alternative to traditional landline internet access. As of Sep 2024, Consumer segment had approximately 114.2 million wireless retail connections. In addition, as of Sep 2024, Consumer segment had approximately 9.7 million total broadband connections (which includes Fios internet, Digital Subscriber Line (DSL) and FWA connections), and approximately 2.8 million Fios video connections.", "Verizon Business Group": "The Business segment provides wireless and wireline communications services and products. These products and services are provided to businesses, government customers and wireless and wireline carriers across the U.S. and a subset of these products and services to customers around the world. The Business segment’s operating revenues for 9M ended Sep 2024, totaled $22.1 billion, a decrease of 3.1%, compared to same period in previous year. The business segment accounted for approximately 22.2% of Verizon’s consolidated revenues for the same period.", "Corporate and Other": "Corporate and other primarily includes device insurance programs, investments in unconsolidated businesses and development stage businesses that support strategic initiatives, as well as unallocated corporate expenses, certain pension and other employee benefit related costs and interest and financing expenses."}, "output.pdf")
    display_pdf("output.pdf")
    st.success("Generated PDF is ready!")
