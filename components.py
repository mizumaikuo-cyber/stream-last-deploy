"""
簡易版の components モジュール（UI 表示）。
・両モードで参照ドキュメントのありかとページ番号を表示
・LLM 応答の形式に依存しないよう、多様なレスポンス形状をハンドリング
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import glob
import streamlit as st
import constants as ct
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def display_app_title():
    st.title("社内情報特化型生成AI検索アプリ")


def display_sidebar_info():
    with st.sidebar:
        st.markdown("### 利用目的\n- 社内文書の検索\n- 社内問い合わせ")
        # Optional diagnostics
        stats = st.session_state.get("ingest_stats")
        if stats:
            st.markdown("---")
            st.markdown("#### 取り込み診断")
            st.caption(
                f"Docs: {stats.get('total_docs','?')} | PDFs: {stats.get('pdf_count','?')} | PDFs with text: {stats.get('pdf_nonempty_count','?')}"
            )
            empties = stats.get("pdf_empty_examples") or []
            if empties:
                st.caption("文字が取れなかったPDF (例):")
                for e in empties:
                    st.caption(f"- {e}")


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
    # Normalize quoted-empty like "" or '' to empty
    if isinstance(text, str) and text.strip() in ('""', "''"):
        text = ""
    # 検索モードでは、関連ありの場合に空文字を返すフローがあるため、空文字時のUXを補強
    if isinstance(text, str) and text.strip() == "":
        if sources:
            msg = "関連する社内文書が見つかりました。下記の参照ドキュメントをご確認ください。"
            st.info(msg)
            _render_sources(sources)
            return msg
        else:
            msg = getattr(ct, "NO_DOC_MATCH_ANSWER", "該当資料なし")
            st.warning(msg)
            return msg

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
                return f"{user_prompt} に対して、従業員一覧を表示しました。"
        # 環境関連の一般フォールバック
        if _looks_like_environment_request(user_prompt):
            _render_environment_fallback(has_sources=bool(sources))
            _render_sources(sources)
            return "一般的な観点による補足情報を提示しました。"
        # 一覧生成に失敗した場合は従来の警告＋参照表示
        st.warning(no_answer_msg)
        _render_sources(sources)
        return no_answer_msg

    # 通常表示
    st.markdown(text)
    _render_sources(sources)
    return text


def detect_dept_listing(prompt: str, dept_name: str = "人事部") -> bool:
    """Public helper to detect department listing intent from user prompt."""
    return _looks_like_dept_listing_request(prompt, dept_name=dept_name)


def render_department_listing_from_data_root(dept_name: str, min_rows: int = 4) -> bool:
    """Render department roster by scanning CSVs under data root, without LLM or retriever.

    Returns True if rendered, False otherwise.
    """
    try:
        data_root = getattr(ct, "RAG_TOP_FOLDER_PATH", "./data")
        pattern = os.path.join(data_root, "**", "*.csv")
        csvs = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if not csvs:
            return False
        # Try pandas first
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # type: ignore

        # Search each CSV for dept rows
        for path in csvs:
            if pd is not None:
                df = None
                for enc in ("utf-8", "utf-8-sig", "cp932"):
                    try:
                        df = pd.read_csv(path, encoding=enc)
                        break
                    except Exception:
                        continue
                if df is None or df.empty:
                    continue
                cols = [str(c) for c in df.columns]
                dept_cols = [c for c in cols if any(k in c for k in ["部署", "部門", "所属", "部"])]
                if dept_cols:
                    mask = False
                    for c in dept_cols:
                        mask = mask | (df[c].astype(str) == dept_name)
                    sub = df.loc[mask]
                else:
                    mask = df.apply(lambda row: row.astype(str).str.contains(dept_name, na=False).any(), axis=1)
                    sub = df.loc[mask]
                if sub is None or sub.empty or len(sub) < min_rows:
                    continue
                st.markdown(f"### {dept_name} の従業員一覧 (CSV 検索)")
                st.dataframe(sub, use_container_width=True)
                return True
            else:
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
                cols = list(rows[0].keys()) if rows else []
                dept_cols = [c for c in cols if any(k in c for k in ["部署", "部門", "所属", "部"])]
                def row_matches(r: dict) -> bool:
                    if dept_cols:
                        for c in dept_cols:
                            if str(r.get(c, "")) == dept_name:
                                return True
                        return False
                    return any(dept_name in str(v) for v in r.values())
                sub_rows = [r for r in rows if row_matches(r)]
                if len(sub_rows) < min_rows:
                    continue
                st.markdown(f"### {dept_name} の従業員一覧 (CSV 検索)")
                table_cols = list({k for r in sub_rows for k in r.keys()})
                if table_cols:
                    header = "| " + " | ".join(table_cols) + " |"
                    sep = "|" + "|".join([" --- "] * len(table_cols)) + "|"
                    st.markdown(header)
                    st.markdown(sep)
                    for r in sub_rows:
                        row = [str(r.get(c, "")) for c in table_cols]
                        st.markdown("| " + " | ".join(row) + " |")
                return True
    except Exception:
        return False
    return False


def render_keyword_search_fallback(query: str, max_hits: int = 5) -> bool:
    """Naive keyword search without embeddings/LLM. Loads docs and filters by substring.

    Returns True if any hits are displayed.
    """
    data_root = getattr(ct, "RAG_TOP_FOLDER_PATH", "./data")
    exts_map = getattr(ct, "SUPPORTED_EXTENSIONS", {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
    })
    lower_q = str(query).strip().lower()
    if not lower_q:
        return False

    def iter_files(root: str):
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                yield os.path.join(dirpath, fn)

    hits: List[Dict[str, Any]] = []
    for path in iter_files(data_root):
        ext = os.path.splitext(path)[1].lower()
        if ext not in exts_map:
            continue
        try:
            loader_ctor = exts_map[ext]
            loader = loader_ctor(path)
            docs = loader.load()
        except Exception:
            continue
        for d in docs:
            text = str(getattr(d, "page_content", ""))
            if lower_q in text.lower():
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source") or meta.get("file_path") or meta.get("path") or meta.get("url") or path
                page = meta.get("page_number") or meta.get("page")
                hits.append({
                    "source": src,
                    "page": page,
                    "snippet": _make_snippet(text),
                })
                if len(hits) >= max_hits:
                    break
        if len(hits) >= max_hits:
            break

    if not hits:
        st.warning("検索フォールバックで一致する資料は見つかりませんでした。")
        return False

    st.info("LLM利用上限のため、簡易検索の結果を表示します。")
    _render_sources(hits)
    return True


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
            page = _extract_page_from_meta(meta)

        # src が本文に直接含まれている場合（保険）
        if src is None and isinstance(item, dict):
            src = item.get("source") or item.get("url") or item.get("path")
            if page is None:
                page = _extract_page_from_meta(item)

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


def _extract_page_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    """Try several common locations/keys for page info and coerce to int if possible."""
    def coerce(v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        # Strings like "5" or "0005"
        try:
            s = str(v).strip()
            if s.isdigit():
                return int(s)
        except Exception:
            pass
        return None

    candidates: List[Any] = []
    # direct keys
    for k in ("page_number", "page", "page_index", "pageIndex", "page_label"):
        if k in meta:
            candidates.append(meta.get(k))
    # nested 'loc'
    loc = meta.get("loc")
    if isinstance(loc, dict):
        for k in ("page_number", "page", "page_index", "pageIndex", "page_label"):
            if k in loc:
                candidates.append(loc.get(k))
    # pick first coercible
    for v in candidates:
        p = coerce(v)
        if p is not None:
            return p
    return None


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
        if display_page is None:
            page_text = " (ページ情報なし)"
        else:
            page_text = f" p.{display_page}"
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


def _looks_like_environment_request(prompt: str) -> bool:
    if not isinstance(prompt, str) or not prompt:
        return False
    keywords = [
        "環境への取り組み",
        "環境",
        "サステナビリティ",
        "ESG",
        "脱炭素",
        "カーボンニュートラル",
        "環境方針",
        "ISO14001",
    ]
    p = prompt
    return any(k in p for k in keywords)


def _render_environment_fallback(has_sources: bool = False) -> None:
    if has_sources:
        st.info("参照資料は見つかりましたが、質問に対して十分な根拠が抽出できなかったため、一般的な観点を補足します。")
    else:
        st.warning("社内の一次資料が見つからなかったため、一般的な観点での回答を提示します。")
    st.markdown(
        """
        #### 環境への取り組みの一般的な観点
        - 方針・ガバナンス: 環境方針の策定・公開、責任体制（経営/管掌役員）の明確化
        - 法令順守と目標設定: 法令順守体制、CO2排出・廃棄物・水資源などの削減目標とKPI
        - 事業活動での削減: 省エネ・再エネ導入、物流最適化、紙使用削減、グリーン購入
        - 認証・評価: ISO 14001 等の取得検討、第三者評価・監査の受審
        - 開示・コミュニケーション: ウェブサイトやレポートでの開示、従業員・取引先への周知

        もし社内の正式な「環境方針」「ESG/サステナビリティ関連資料」が存在する場合、
        データフォルダに追加いただくと、以後は社内資料に基づく回答が可能になります。
        """
    )


def _try_render_department_listing(sources: List[Dict[str, Any]], dept_name: str, min_rows: int = 4) -> bool:
    # 参照リストから CSV を優先して探す
    csv_paths: List[str] = []
    for s in sources:
        src = s.get("source")
        if isinstance(src, str) and src.lower().endswith(".csv"):
            # デプロイ環境では絶対パスが無効な場合があるため、ローカル data 配下を探索
            resolved = _resolve_local_data_path(src)
            if resolved:
                csv_paths.append(resolved)
            else:
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
            # Markdown テーブルで描画（pandas 依存を避ける）
            cols = list({k for r in sub_rows for k in r.keys()})
            if cols:
                header = "| " + " | ".join(cols) + " |"
                sep = "|" + "|".join([" --- "] * len(cols)) + "|"
                st.markdown(header)
                st.markdown(sep)
                for r in sub_rows:
                    row = [str(r.get(c, "")) for c in cols]
                    st.markdown("| " + " | ".join(row) + " |")
            return True

    return False


def _resolve_local_data_path(path: str) -> Optional[str]:
    try:
        # 既に存在する
        if os.path.exists(path):
            return path
        # ファイル名で data/ 以下を探索
        base = os.path.basename(path)
        data_root = getattr(ct, "RAG_TOP_FOLDER_PATH", "./data")
        pattern = os.path.join(data_root, "**", base)
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            if os.path.isfile(m):
                return m
    except Exception:
        return None
    return None
