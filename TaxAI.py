import streamlit as st
import logging
from chatbot_logic import generate_answer

# Tiêu đề và phần mô tả
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .description {
        font-size: 18px;
        color: #f5f2f2;
        text-align: center;
    }
    .note {
        font-size: 20px;
        color: #FF5722;
        text-align: center;
    }
    </style>
    <div class="title">Trợ lý AI Tư Vấn về Chính Sách Pháp Luật Thuế</div>
    <div class="note">Các câu hỏi mang tính chất tham khảo, không mang tính chất đại diện cho cơ quan Thuế giải đáp thắc mắc.</div>
    """, unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Hiển thị lịch sử tin nhắn
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý câu hỏi người dùng
question = st.chat_input("Nhập câu hỏi của bạn")

if question:
    # Ghi lại câu hỏi
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Tạo câu trả lời
    answer = generate_answer(question)

    # Hiển thị câu trả lời
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Ghi lại câu trả lời
    st.session_state["messages"].append({"role": "assistant", "content": answer})
