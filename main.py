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

    # Off-topic guard: skip LLM/retriever calls and show fixed message per mode
    try:
        if hasattr(cp, "_looks_unrelated_to_corp_docs") and cp._looks_unrelated_to_corp_docs(chat_message, []):
            if mode == ct.ANSWER_MODE_1:
                fixed = "入力内容と関連する社内文書が見つかりませんでした"
            else:
                fixed = getattr(ct, "INQUIRY_NO_MATCH_ANSWER", "回答に必要な情報が見つかりませんでした。")
            with st.chat_message("assistant"):
                st.warning(fixed)
            try:
                st.session_state.messages.append({"role": "assistant", "content": fixed})
            except Exception:
                pass
            st.stop()

        # Normal LLM flow
        llm_resp = utils.get_llm_response(chat_message)
    except Exception as e:
        # If quota error and the intent is department listing, try CSV direct fallback
        err_msg = str(e)
        # If off-topic (e.g., 天気/雑談) and quota error, prefer fixed message over fallbacks
        if hasattr(cp, "_looks_unrelated_to_corp_docs") and cp._looks_unrelated_to_corp_docs(chat_message, []):
            with st.chat_message("assistant"):
                if mode == ct.ANSWER_MODE_1:
                    fixed = "入力内容と関連する社内文書が見つかりませんでした"
                else:
                    fixed = getattr(ct, "INQUIRY_NO_MATCH_ANSWER", "回答に必要な情報が見つかりませんでした。")
                st.warning(fixed)
            st.stop()
        # Dept listing offline fallback
        if cp.detect_dept_listing(chat_message, dept_name=None) and (
            "insufficient_quota" in err_msg or "You exceeded your current quota" in err_msg or "Error code: 429" in err_msg
        ):
            with st.chat_message("assistant"):
                # infer department name
                dept = cp.get_department_from_prompt(chat_message) or "人事部"
                rendered = cp.render_department_listing_from_data_root(dept, min_rows=4)
                if not rendered:
                    st.warning("部署一覧のCSV検索に失敗しました。")
            st.stop()
        # If quota error for general query, try naive keyword search fallback
        if "insufficient_quota" in err_msg or "You exceeded your current quota" in err_msg or "Error code: 429" in err_msg:
            with st.chat_message("assistant"):
                rendered = cp.render_keyword_search_fallback(chat_message, max_hits=5)
                if not rendered:
                    st.error(utils.build_error_message(f"エラーが発生しました: {e}"))
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
        # Fallback to a short placeholder when content is empty
        safe_content = content if isinstance(content, str) and content.strip() else "参照ドキュメントを表示しました。"
        st.session_state.messages.append({"role": "assistant", "content": safe_content})
    except Exception:
        pass