import os
import logging
import requests
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Kiểm tra các biến môi trường thiết yếu
if not os.getenv("OLLAMA_MODEL") or not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
    logging.error("Thiếu biến môi trường thiết yếu!")
    exit(1)  # Dừng chương trình nếu thiếu biến môi trường thiết yếu

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("streamlit_app.log", mode="w")],
)

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Đường dẫn đến thư mục vectorstore
persist_directory = "vectorstore/chroma_db"

# Khởi tạo embeddings từ Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# Di chuyển mô hình embeddings lên GPU nếu có GPU
embeddings.model.to(device)

# Load Vectorstore từ thư mục
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="rag-chroma"
)

# Khởi tạo mô hình Ollama
ollama = OllamaLLM(model=os.getenv("OLLAMA_MODEL"))

# Nếu Ollama hỗ trợ GPU, di chuyển mô hình lên GPU (kiểm tra tài liệu Ollama về cách sử dụng GPU)
ollama.model.to(device)

def web_search(query: str) -> str:
    """Tìm kiếm trên web qua Google CSE API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        return "Không thể tìm kiếm trên web do thiếu API Key hoặc CSE ID."

    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()

        results = response.json().get("items", [])
        if not results:
            return "Không tìm thấy kết quả nào trên web."

        # Chỉ trích xuất đường dẫn từ các kết quả đầu tiên
        links = [item.get("link", "#") for item in results[:3]]  # Lấy 3 kết quả đầu tiên
        return "\n".join(links)
    except requests.exceptions.RequestException as e:
        logging.error(f"Lỗi khi tìm kiếm trên web: {e}")
        return "Lỗi xảy ra khi tìm kiếm trên web."

def generate_answer(question: str) -> str:
    """Tạo câu trả lời dựa trên câu hỏi người dùng."""
    sources = []

    # Tìm kiếm từ Vectorstore
    try:
        results = vectorstore.similarity_search(question, k=3)
        if results:
            context = "\n".join([result.page_content for result in results])
            sources.append("**Nguồn VectorDB:**\n" + "\n".join([result.metadata.get("source", "Không rõ nguồn") for result in results]))
            logging.info(f"Context retrieved from VectorDB: {context}")
        else:
            context = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
    except Exception as e:
        logging.error(f"Error during similarity search: {e}")
        context = "Không thể truy xuất thông tin từ VectorDB."

    # Tìm kiếm trên web
    try:
        web_results = web_search(question)
        sources.append("**Nguồn Web Search:**\n" + web_results)
    except Exception as e:
        logging.error(f"Error during web search: {e}")
        web_results = "Không thể tìm kiếm trên web."

    # Gộp kết quả từ VectorDB và Web Search
    combined_context = f"{context}\n\n{web_results}"

    # Tạo câu trả lời từ mô hình Ollama
    try:
        prompt = f"Câu hỏi: {question}\nBối cảnh:\n{combined_context}\nCâu trả lời:"
        response = ollama.generate([prompt])
        answer = response.generations[0][0].text.strip()
        logging.info(f"Assistant response: {answer}")
    except Exception as e:
        logging.error(f"Error during LLM response generation: {e}")
        answer = "Xin lỗi, hiện tại hệ thống không thể xử lý câu hỏi của bạn."

    # Thêm trích dẫn nguồn
    if sources:
        return f"{answer}\n\n**Trích dẫn nguồn:**\n" + "\n\n".join(sources)
    else:
        return f"{answer}\n\n**Không có nguồn trích dẫn.**"
