"""
簡易版の components モジュール（UI 表示）。
・両モードで参照ドキュメントのありかとページ番号を表示
・LLM 応答の形式に依存しないよう、多様なレスポンス形状をハンドリング
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import streamlit as st
import constants as ct


def display_app_title():
    st.title("社内情報特化型生成AI検索アプリ")


def display_sidebar_info():
    with st.sidebar:
        st.markdown("### 利用目的\n- 社内文書の検索\n- 社内問い合わせ")


def display_select_mode():
    if "mode" not in st.session_state:
        st.session_state.mode = "社内文書検索"
    st.sidebar.radio(
        "モード",
        ["社内文書検索", "社内問い合わせ"],
        key="mode",
        horizontal=False,
    )


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
    text, sources = _extract_answer_and_sources(resp)
    # 検索モードでは、関連ありの場合に空文字を返すフローがあるため、空文字時のUXを補強
    if isinstance(text, str) and text.strip() == "":
        if sources:
            st.info("関連する社内文書が見つかりました。下記の参照ドキュメントをご確認ください。")
            _render_sources(sources)
        else:
            st.warning(getattr(ct, "NO_DOC_MATCH_ANSWER", "該当資料なし"))
        return text

    st.markdown(text)
    _render_sources(sources)
    return text


def display_contact_llm_response(resp):
    text, sources = _extract_answer_and_sources(resp)
    # 問い合わせモードでも万一空文字ならフォールバック
    if isinstance(text, str) and text.strip() == "":
        st.warning(getattr(ct, "INQUIRY_NO_MATCH_ANSWER", "回答に必要な情報が見つかりませんでした。"))
        _render_sources(sources)
        return text

    st.markdown(text)
    _render_sources(sources)
    return text


# =========================
# internal helpers
# =========================

def _extract_answer_and_sources(resp: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    レスポンスから回答テキストと参照ドキュメント情報を抽出する。

    対応する代表的な形状:
      - LangChainの返却オブジェクトに近い: resp.answer / resp.source_documents
      - dict 形: {"answer"|"content"|"text", "source_documents"|"sources"|"context"}
      - フォールバック: str(resp)
    各ドキュメントの metadata から以下を優先的に読む:
      source|file_path|path|url, page_number|page
    """
    # 回答テキスト
    text: str = (
        getattr(resp, "answer", None)
        or getattr(resp, "content", None)
        or getattr(resp, "text", None)
        or getattr(resp, "output_text", None)
        or getattr(resp, "result", None)
        or getattr(resp, "message", None)
        or getattr(resp, "response", None)
        or (resp.get("answer") if isinstance(resp, dict) else None)
        or (resp.get("content") if isinstance(resp, dict) else None)
        or (resp.get("text") if isinstance(resp, dict) else None)
        or (resp.get("output_text") if isinstance(resp, dict) else None)
        or (resp.get("result") if isinstance(resp, dict) else None)
        or (resp.get("message") if isinstance(resp, dict) else None)
        or (resp.get("response") if isinstance(resp, dict) else None)
        or str(resp)
    )

    # ソース候補の収集
    raw_sources: List[Any] = []
    # 属性から
    if hasattr(resp, "source_documents"):
        try:
            raw = getattr(resp, "source_documents")
            if isinstance(raw, Iterable):
                raw_sources.extend(list(raw))
        except Exception:
            pass
    # dict キーから
    if isinstance(resp, dict):
        for key in ("source_documents", "sources", "context", "documents", "docs", "relevant_docs", "retrieved_docs"):
            if key in resp and isinstance(resp[key], Iterable) and not isinstance(resp[key], (str, bytes)):
                try:
                    raw_sources.extend(list(resp[key]))
                except Exception:
                    pass

    # Document/辞書から正規化
    normalized: List[Dict[str, Any]] = []
    for item in raw_sources:
        src = None
        page: Optional[int] = None
        meta = None

        # LangChain Document 風: .metadata
        if hasattr(item, "metadata"):
            try:
                meta = getattr(item, "metadata")
            except Exception:
                meta = None
        # dict 風
        if meta is None and isinstance(item, dict):
            meta = item.get("metadata", item)

        if isinstance(meta, dict):
            src = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("path")
                or meta.get("url")
            )
            page = meta.get("page_number", meta.get("page"))

        # src が本文に直接含まれている場合（保険）
        if src is None and isinstance(item, dict):
            src = item.get("source") or item.get("url") or item.get("path")
            page = item.get("page_number", item.get("page"))

        if src:
            normalized.append({"source": src, "page": page})

    # 重複排除
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for s in normalized:
        key = (s["source"], s.get("page"))
        if key not in seen:
            seen.add(key)
            dedup.append(s)

    # Top-Kで丸め（定数があれば）
    try:
        k = int(getattr(ct, "RETRIEVAL_TOP_K", len(dedup)))
        dedup = dedup[:k]
    except Exception:
        pass

    return text, dedup


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.caption(":grey[参照ドキュメントはありません]")
        return

    st.markdown("---")
    st.markdown("#### 参照ドキュメント")
    for s in sources:
        src = s.get("source")
        page = s.get("page")
        if not src:
            continue
        is_link = isinstance(src, str) and src.startswith(("http://", "https://"))
        icon = ct.LINK_SOURCE_ICON if is_link else ct.DOC_SOURCE_ICON
        # 0 始まりのページ番号に対処（0 の時だけ 1 として表示）
        if isinstance(page, int) and page == 0:
            display_page = 1
        else:
            display_page = page
        page_text = f" p.{display_page}" if display_page is not None else ""
        st.markdown(f"- {icon}{src}{page_text}")
