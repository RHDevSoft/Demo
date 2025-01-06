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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

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

            # Template to guide the AI in generating structured data
            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""Given the following context, extract the following structured information and organize it into these sections:
1. Business Profile
2. Operating Segments
3. Verizon Consumer Group
4. Verizon Business Group
5. Corporate and Other

Context:
{context}

Ensure the content is organized under the correct section and formatted properly for a structured document. Your response should follow the format below:

Business Profile:
<Extracted information for Business Profile>

Operating Segments:
<Extracted information for Operating Segments>

Verizon Consumer Group:
<Extracted information for Verizon Consumer Group>

Verizon Business Group:
<Extracted information for Verizon Business Group>

Corporate and Other:
<Extracted information for Corporate and Other>
"""
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
        
        # Join all responses into a single string and return
        return "\n\n".join(responses)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def generate_pdf(data, output_path):
    # Create a SimpleDocTemplate to handle PDF generation with proper formatting
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    # Use a default style for the PDF
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    for section, content in data.items():
        # Add section title as a heading
        section_paragraph = Paragraph(f"<b>{section}</b>", normal_style)
        story.append(section_paragraph)
        
        # Add content with line breaks
        content_paragraph = Paragraph(content, normal_style)
        story.append(content_paragraph)
        story.append(Paragraph("<br/>", normal_style))  # Add some space between sections

    # Build the PDF
    doc.build(story)

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

# Add query input to dynamically generate content
query = st.text_input("Enter your query to extract information")
if query:
    with st.spinner("Processing your query..."):
        result = asyncio.run(handle_query(query))
        if result:
            # Parse the AI response into sections (Business Profile, Operating Segments, etc.)
            sections = {}
            for section in ["Business Profile", "Operating Segments", "Verizon Consumer Group", "Verizon Business Group", "Corporate and Other"]:
                start_index = result.find(section)
                if start_index != -1:
                    end_index = result.find("\n", start_index + len(section))
                    content = result[start_index + len(section):end_index].strip() if end_index != -1 else result[start_index + len(section):]
                    sections[section] = content.strip()
            
            # Generate the PDF with AI-generated content
            generate_pdf(sections, "generated_output.pdf")
            display_pdf("generated_output.pdf")
            st.success("Generated PDF is ready!")
