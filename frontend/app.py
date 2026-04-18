import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

# 🔥 Page config
st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🤖",
    layout="wide"
)

# 🔥 Custom CSS (modern UI)
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    .user-msg {
        background-color: #1e293b;
        color: white;
        border-radius: 12px;
        padding: 12px;
    }

    .bot-msg {
        background-color: #0f172a;
        color: #e2e8f0;
        border-radius: 12px;
        padding: 12px;
    }

    .title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .stButton>button {
        border-radius: 10px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# 🔥 Header
st.markdown('<div class="title">🚀 Enterprise RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything from your knowledge base</div>', unsafe_allow_html=True)

# 🔥 Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔥 Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# 🔥 Input
query = st.chat_input("Ask your question...")

if query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)

    # 🔥 API call
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response")

                    st.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:
                    st.error("API Error")

            except Exception as e:
                st.error("Failed to connect to backend")