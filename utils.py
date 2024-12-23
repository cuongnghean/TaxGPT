import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Đường dẫn đến thư mục Data trong Google Drive
folder_path = 'Data'

def load_pdf_documents():
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    all_data = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        all_data.extend(data)
    return all_data

def split_documents_into_chunks(all_data, chunk_size=1024, chunk_overlap=62):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(all_data)
