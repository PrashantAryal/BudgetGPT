from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_budget_speech_txt_np(TXT_PATH_NP, chunk_size=500, chunk_overlap=40):
    """
    Extracts text from a text file (budget speech) and splits it into chunks for indexing.
    :param txt_path: Path to the text file
    :param chunk_size: The maximum size of each chunk of text
    :param chunk_overlap: The overlap between chunks to ensure context is maintained
    :return: A list of Document objects
    """
    # Read the text file
    with open(TXT_PATH_NP, 'r', encoding='utf-8') as file:
        full_text = ""
        full_text = file.read()  
    
    
    # Initialize the text splitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the full text into smaller chunks
    chunks = text_splitter.split_text(full_text)
    
    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
   
    return documents


def parse_budget_speech_pdf_en(PDF_PATH, chunk_size=500, chunk_overlap=40):
    """
    Extracts text from a PDF document (budget speech) and splits it into chunks for indexing.
    
    :param pdf_path: Path to the PDF file
    :param chunk_size: The maximum size of each chunk of text
    :param chunk_overlap: The overlap between chunks to ensure context is maintained
    :return: A list of Document objects
    """
    # Read the PDF
    reader = PdfReader(PDF_PATH)
    full_text = ""
    
    # Extract text from each page
    for page in reader.pages:
        full_text += page.extract_text()
    
    # Initialize the text splitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the full text into smaller chunks
    chunks = text_splitter.split_text(full_text)
    
    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents