"""
簡易版の components モジュール（main.py の参照を満たす最小機能を提供）
"""
import streamlit as st


def display_app_title():
    st.title("社内情報特化型生成AI検索アプリ")


def display_sidebar_info():
    with st.sidebar:
        st.markdown("### 利用目的\n- 社内文書の検索\n- 社内問い合わせ")


def display_select_mode():
    if "mode" not in st.session_state:
        st.session_state.mode = "社内文書検索"
    st.sidebar.selectbox("モード", ["社内文書検索", "社内問い合わせ"], key="mode")


def display_initial_ai_message():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "どうぞ質問してください。"})


def display_conversation_log():
    for msg in st.session_state.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def display_search_llm_response(resp):
    text = getattr(resp, "text", str(resp))
    st.write(text)
    return text


def display_contact_llm_response(resp):
    text = getattr(resp, "text", str(resp))
    st.write(text)
    return text
