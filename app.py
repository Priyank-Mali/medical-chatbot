import streamlit as st
from chatbot import get_bot_response

st.set_page_config(page_title="Priyank Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask any medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_bot_response(user_input)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
