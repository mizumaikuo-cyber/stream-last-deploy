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
    user_prompt = _last_user_prompt()

    # 返答が空、または「見つかりません」固定文言の場合のフォールバック
    no_answer_msg = getattr(ct, "INQUIRY_NO_MATCH_ANSWER", "回答に必要な情報が見つかりませんでした。")
    is_empty = isinstance(text, str) and text.strip() == ""
    is_no_answer = isinstance(text, str) and text.strip() == no_answer_msg

    if (is_empty or is_no_answer) and sources:
        # 人事部の一覧要求に対しては roster CSV から一覧を生成して提示
        if _looks_like_dept_listing_request(user_prompt, dept_name="人事部"):
            rendered = _try_render_department_listing(sources, dept_name="人事部", min_rows=4)
            if rendered:
                _render_sources(sources)
                return text
        # 一覧生成に失敗した場合は従来の警告＋参照表示
        st.warning(no_answer_msg)
        _render_sources(sources)
        return text

    # 通常表示
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
        content: Optional[str] = None

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

        # 抜粋用の本文テキスト
        try:
            if hasattr(item, "page_content"):
                content = getattr(item, "page_content")
            elif isinstance(item, dict):
                content = item.get("page_content") or item.get("content") or item.get("text")
        except Exception:
            content = None

        if src:
            snippet = _make_snippet(content)
            normalized.append({"source": src, "page": page, "snippet": snippet})

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
        snippet = s.get("snippet")
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
        if isinstance(snippet, str) and snippet.strip():
            with st.expander("抜粋を表示", expanded=False):
                st.markdown(
                    "> " + snippet.replace("\n", " ").strip()
                )


def _make_snippet(text: Optional[str], limit: int = 280) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip()
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _last_user_prompt() -> str:
    try:
        msgs = st.session_state.get("messages", [])
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
    except Exception:
        pass
    return ""


def _looks_like_dept_listing_request(prompt: str, dept_name: str) -> bool:
    if not isinstance(prompt, str) or not prompt:
        return False
    p = prompt
    if dept_name not in p:
        return False
    # 「一覧」「リスト」「列挙」などの語と、社員/従業員などの語が含まれるかを簡易チェック
    want_list = any(k in p for k in ["一覧", "リスト", "列挙", "まとめ", "一覧化"])
    employee_word = any(k in p for k in ["従業員", "社員", "メンバー", "人物", "人員"])
    return want_list and employee_word


def _try_render_department_listing(sources: List[Dict[str, Any]], dept_name: str, min_rows: int = 4) -> bool:
    # 参照リストから CSV を優先して探す
    csv_paths: List[str] = []
    for s in sources:
        src = s.get("source")
        if isinstance(src, str) and src.lower().endswith(".csv"):
            csv_paths.append(src)
    # 見つからない場合は失敗
    if not csv_paths:
        return False

    # pandas は遅延 import（無い環境でも UI は壊さない）。無ければ csv モジュールで代替。
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    for path in csv_paths:
        if pd is not None:
            # pandas あり
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except Exception:
                # 一部 CSV は BOM 付きなどのためリトライ
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                except Exception:
                    try:
                        df = pd.read_csv(path, encoding="cp932")
                    except Exception:
                        df = None

            if df is None or df.empty:
                continue

            # 部署っぽいカラムを推測
            cols = [str(c) for c in df.columns]
            dept_cols = [c for c in cols if any(k in c for k in ["部署", "部門", "所属", "部"]) ]
            if not dept_cols:
                # カラム名が読めない場合、全文検索的に行フィルタ
                mask = df.apply(lambda row: row.astype(str).str.contains(dept_name, na=False).any(), axis=1)
                sub = df.loc[mask]
            else:
                # いずれかの部署カラムが dept_name と一致する行を抽出
                mask = False
                for c in dept_cols:
                    mask = mask | (df[c].astype(str) == dept_name)
                sub = df.loc[mask]

            if sub is None or sub.empty:
                continue

            # 4行以上出ることが条件。満たさない場合は次の CSV にトライ
            if len(sub) < min_rows:
                continue

            st.markdown(f"### {dept_name} の従業員一覧")
            st.dataframe(sub, use_container_width=True)
            return True
        else:
            # pandas なし: csv モジュールで読み込み
            import csv
            rows: List[dict] = []
            read_ok = False
            for enc in ("utf-8", "utf-8-sig", "cp932"):
                try:
                    with open(path, "r", encoding=enc, newline="") as f:
                        reader = csv.DictReader(f)
                        rows = [r for r in reader]
                        read_ok = True
                        break
                except Exception:
                    continue
            if not read_ok or not rows:
                continue

            # 列名を収集
            cols = list(rows[0].keys()) if rows else []
            dept_cols = [c for c in cols if any(k in c for k in ["部署", "部門", "所属", "部"]) ]

            def row_matches(r: dict) -> bool:
                if dept_cols:
                    for c in dept_cols:
                        if str(r.get(c, "")) == dept_name:
                            return True
                    return False
                # 部署カラムが特定できない場合は全列に対して包含検索
                return any(dept_name in str(v) for v in r.values())

            sub_rows = [r for r in rows if row_matches(r)]
            if len(sub_rows) < min_rows:
                continue

            st.markdown(f"### {dept_name} の従業員一覧")
            st.table(sub_rows)
            return True

    return False
