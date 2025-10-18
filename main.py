"""
Streamlit app entrypoint. Clean, ASCII-safe wiring.
"""

import streamlit as st

import components as cp
import initialize as init
import utils
import constants as ct


st.set_page_config(page_title=ct.APP_NAME, page_icon="🔎")
st.title(ct.APP_NAME)

# Initialize session, retriever, logger
try:
    init.initialize()
except Exception as e:
    st.error(utils.build_error_message(f"初期化処理に失敗しました: {e}"))
    st.stop()

# Sidebar info and controls
cp.display_sidebar_info()
# Sidebar controls: create widget and read its value from session_state
cp.display_select_mode()
mode = st.session_state.get("mode", ct.ANSWER_MODE_1)

# Conversation log
cp.display_conversation_log()

# Input and response
chat_message = st.chat_input("質問を入力してください…")
if chat_message:
    # Ensure message log exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("user"):
        st.markdown(chat_message)
    # Append to conversation log (for components' fallback detection)
    st.session_state.messages.append({"role": "user", "content": chat_message})

    try:
        llm_resp = utils.get_llm_response(chat_message)
    except Exception as e:
        # If quota error and the intent is department listing, try CSV direct fallback
        err_msg = str(e)
        if cp.detect_dept_listing(chat_message, dept_name="人事部") and (
            "insufficient_quota" in err_msg or "You exceeded your current quota" in err_msg or "Error code: 429" in err_msg
        ):
            with st.chat_message("assistant"):
                rendered = cp.render_department_listing_from_data_root("人事部", min_rows=4)
                if not rendered:
                    st.warning("部署一覧のCSV検索に失敗しました。")
            st.stop()
        # Otherwise, show normal error
        st.error(utils.build_error_message(f"エラーが発生しました: {e}"))
        st.stop()

    with st.chat_message("assistant"):
        if mode == ct.ANSWER_MODE_1:
            content = cp.display_search_llm_response(llm_resp)
        else:
            content = cp.display_contact_llm_response(llm_resp)
    # Append assistant message to log
    try:
        st.session_state.messages.append({"role": "assistant", "content": content})
    except Exception:
        pass