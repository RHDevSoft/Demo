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

async def generate_structured_pdf():
    try:
        responses = []
        # Iterate through the PDF data and extract information
        for filename, vector_store in st.session_state.pdf_data.items():
            vector_store = FAISS.load_local(
                os.path.join(PERSIST_DIR, filename),
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
            )
            docs = vector_store.similarity_search("Please extract the full details for each section: Business Profile, Operating Segments, Verizon Consumer Group, Verizon Business Group, Corporate and Other.")  # More detailed query
            context = "\n".join([doc.page_content for doc in docs])

            # Template to guide the AI in generating structured data
            prompt_template = PromptTemplate(
                input_variables=["context"],
                template="""Given the following context, extract detailed information under the following sections and organize it clearly:
1. Business Profile
2. Operating Segments
3. Verizon Consumer Group
4. Verizon Business Group
5. Corporate and Other

Context:
{context}

For each section, ensure that you extract and include **all relevant details** under the appropriate heading, such as company overview, financials, and key metrics. Format the information neatly and clearly, with each section clearly labeled and all important details included.
"""
            )
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = await chain.arun(context=context)
            responses.append(response)

        # Combine all the responses and generate the PDF
        content = "\n\n".join(responses)
        sections = {}
        for section in ["Business Profile", "Operating Segments", "Verizon Consumer Group", "Verizon Business Group", "Corporate and Other"]:
            start_index = content.find(section)
            if start_index != -1:
                end_index = content.find("\n", start_index + len(section))
                content_section = content[start_index + len(section):end_index].strip() if end_index != -1 else content[start_index + len(section):]
                sections[section] = content_section.strip()

        # Generate the PDF
        generate_pdf(sections, "generated_output.pdf")
        display_pdf("generated_output.pdf")
        st.success("Generated PDF is ready!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

def generate_pdf(data, output_path):
    # Create a SimpleDocTemplate to handle PDF generation with proper formatting
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    # Use a default style for the PDF
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    # Set max width for text
    max_width = 450  # Maximum width for text

    # Iterate through the sections and add them to the PDF
    for section, content in data.items():
        # Add section title as a heading
        section_paragraph = Paragraph(f"<b>{section}</b>", normal_style)
        story.append(section_paragraph)

        # Ensure content is wrapped correctly
        wrapped_content = content.replace("\n", "<br/>")
        content_paragraph = Paragraph(wrapped_content, normal_style)

        # Add content to story
        story.append(content_paragraph)
        story.append(Paragraph("<br/>", normal_style))  # Add space between sections

    # Build the PDF
    doc.build(story)

st.title("PDF Processing Chatbot")
st.markdown("Upload PDFs and automatically generate a structured PDF format.")

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
            # Automatically generate the PDF after ingestion
            asyncio.run(generate_structured_pdf())
