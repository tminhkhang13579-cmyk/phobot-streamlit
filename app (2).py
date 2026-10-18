import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ---------------- CẤU HÌNH ỨNG DỤNG ----------------
st.set_page_config(
    page_title="PhoBot - Trợ lý ẩm thực",
    page_icon="🍜",
    layout="centered"
)

MODEL_NAME = "gemini-3.6-flash"

# Lấy API key từ Streamlit Secrets hoặc biến môi trường
try:
    API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    API_KEY = ""

API_KEY = API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ---------------- ĐỌC DỮ LIỆU ----------------
menu_path = Path("menu.csv")
config_path = Path("config.json")

if not menu_path.exists():
    st.error("Không tìm thấy menu.csv. Hãy tải file này lên GitHub.")
    st.stop()

if not config_path.exists():
    st.error("Không tìm thấy config.json. Hãy tải file này lên GitHub.")
    st.stop()

try:
    menu_df = pd.read_csv(menu_path).fillna("")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except Exception as e:
    st.error(f"Không đọc được dữ liệu: {e}")
    st.stop()

# ---------------- CHUẨN BỊ NGỮ CẢNH ----------------
def build_menu_context(df):
    lines = []

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        desc = str(row.get("description", "")).strip()
        ingredients = str(row.get("ingredients", "")).strip()
        notes = str(row.get("notes", "")).strip()

        line = f"- {name}: {desc}"
        if ingredients:
            line += f" Thành phần: {ingredients}."
        if notes:
            line += f" Ghi chú: {notes}."

        lines.append(line)

    return "\n".join(lines)


menu_context = build_menu_context(menu_df)

restaurant_name = config.get("restaurant_name", "Viet Cuisine")
restaurant_address = config.get("restaurant_address", "")
hotline = config.get("hotline", "")
functions = ", ".join(
    config.get("functions", ["tư vấn món ăn"])
)
out_of_scope_message = config.get(
    "out_of_scope_message",
    "Xin lỗi, tôi chưa hỗ trợ câu hỏi này. "
    "Vui lòng liên hệ nhân viên nhà hàng để được trợ giúp."
)

system_instruction = f"""
Bạn tên là PhoBot, trợ lý AI của nhà hàng {restaurant_name}.
Địa chỉ: {restaurant_address}.
Hotline: {hotline}.
Nhiệm vụ: {functions}.

Quy tắc:
1. Luôn trả lời bằng tiếng Việt, thân thiện và lịch sự.
2. Chỉ sử dụng thông tin trong dữ liệu thực đơn được cung cấp.
3. Không tự bịa món ăn, giá tiền, chương trình khuyến mãi
   hoặc thông tin chưa có trong dữ liệu.
4. Nếu không có thông tin được hỏi, hãy nói rõ rằng dữ liệu
   hiện chưa cung cấp thông tin đó.
5. Với câu hỏi ngoài phạm vi nhà hàng và ẩm thực, hãy trả lời:
   "{out_of_scope_message}"
6. Trả lời súc tích, dễ hiểu.
"""

# ---------------- KẾT NỐI GEMINI ----------------
@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


def generate_answer(question):
    if not API_KEY:
        return (
            "⚠️ Ứng dụng chưa được cấu hình Gemini API Key. "
            "Chủ ứng dụng cần thêm GEMINI_API_KEY trong "
            "Settings → Secrets trên Streamlit Cloud."
        )

    try:
        client = get_client(API_KEY)

        prompt = f"""
Thông tin thực đơn:
{menu_context}

Câu hỏi của khách hàng:
{question}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )

        if response.text:
            return response.text

        return "Xin lỗi, tôi chưa tạo được câu trả lời. Bạn thử hỏi lại nhé."

    except Exception as e:
        # Ghi lỗi để chủ ứng dụng xem trong Cloud logs
        print(f"Lỗi Gemini API: {e}")
        return (
            "Xin lỗi, hiện tại tôi chưa thể xử lý yêu cầu. "
            "Bạn vui lòng thử lại sau."
        )


# ---------------- GIAO DIỆN CHAT ----------------
st.title("🍜 PhoBot - Trợ lý ẩm thực")
st.caption(f"Chào mừng bạn đến với {restaurant_name}!")

with st.sidebar:
    st.subheader("🏠 Thông tin nhà hàng")
    st.write(f"**Tên:** {restaurant_name}")

    if restaurant_address:
        st.write(f"**Địa chỉ:** {restaurant_address}")

    if hotline:
        st.write(f"**Hotline:** {hotline}")

    st.write(f"**Số món trong dữ liệu:** {len(menu_df)}")

    if st.button("🗑️ Xóa cuộc trò chuyện"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Xin chào! Tôi là PhoBot của {restaurant_name}. "
                "Bạn muốn tìm hiểu món ăn nào hôm nay?"
            )
        }
    ]

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận câu hỏi từ người dùng
question = st.chat_input("Nhập câu hỏi về thực đơn...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("PhoBot đang suy nghĩ..."):
            answer = generate_answer(question)
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
