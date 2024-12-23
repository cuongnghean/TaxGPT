import os
import logging
from utils import load_pdf_documents, split_documents_into_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,  # Mức log: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Ghi log ra console
        logging.FileHandler("embedding.log", mode="w")  # Ghi log vào file
    ]
)

# Tạo thư mục vectorstore nếu chưa tồn tại
persist_directory = "vectorstore/chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    logging.info(f"Tạo thư mục lưu trữ vectorstore tại: {persist_directory}")

# Tải tài liệu PDF
logging.info("Bắt đầu tải tài liệu PDF...")
all_data = load_pdf_documents()
logging.info(f"Đã tải {len(all_data)} tài liệu PDF.")

# Chia nhỏ văn bản
logging.info("Bắt đầu chia nhỏ văn bản...")
all_splits = split_documents_into_chunks(all_data)
logging.info(f"Đã chia nhỏ tài liệu thành {len(all_splits)} đoạn văn bản.")

# Tạo embeddings
logging.info("Bắt đầu tạo embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
logging.info("Đã khởi tạo mô hình embeddings HuggingFace.")

# Thêm vào cơ sở dữ liệu vector Chroma
logging.info("Bắt đầu thêm văn bản vào cơ sở dữ liệu vector Chroma...")
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory=persist_directory,
)
logging.info("Hoàn thành thêm dữ liệu vào cơ sở dữ liệu vector Chroma.")

# Lưu cơ sở dữ liệu
logging.info("Bắt đầu lưu cơ sở dữ liệu vector...")
vectorstore.persist()
logging.info(f"Cơ sở dữ liệu vector đã được lưu tại: {persist_directory}")

logging.info("Quá trình hoàn tất!")

