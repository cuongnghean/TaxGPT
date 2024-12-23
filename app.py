from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from chatbot_logic import generate_answer  # Import logic của bạn để sinh câu trả lời

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

# Tạo FastAPI app
app = FastAPI()

# Tạo schema cho dữ liệu đầu vào (câu hỏi)
class Question(BaseModel):
    question: str

# API trả lời câu hỏi
@app.post("/api/answer")
async def answer(data: Question):
    question = data.question

    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    # Gọi hàm tạo câu trả lời từ chatbot_logic
    answer = generate_answer(question)
    
    logging.info(f"Received question: {question}, Answer: {answer}")
    return {"answer": answer}

# Chạy FastAPI app
